"""Định danh thực thể chuẩn đặt trên các tracker ID có vòng đời ngắn."""

from dataclasses import dataclass, field
import math

from src.core.analytics.config import IdentityConfig, LabelsConfig
from src.core.analytics.geometry import (
    bbox_center,
    bbox_from_relative_offset,
    bbox_size,
    relative_bbox_offset,
    translate_bbox,
)
from src.core.analytics.types import BagPersonRelation, EntityObservation
from src.core.types import BBox, TrackedObject


@dataclass
class _EntityRecord:
    entity_id: int
    label: str
    class_id: int
    last_bbox: BBox
    last_seen_ms: float
    last_update_ms: float
    last_frame_idx: int
    confidence: float
    reliable: bool
    raw_track_ids: set[int] = field(default_factory=set)
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass(frozen=True)
class _BagAnchor:
    holder_person_id: int
    relative_offset: tuple[float, float, float, float]


class EntityIdentityManager:
    """Reconnect raw tracks while preserving observed versus inferred state."""

    def __init__(self, config: IdentityConfig, labels: LabelsConfig):
        self.config = config
        self.labels = labels
        self._next_entity_id = 1
        self._records: dict[int, _EntityRecord] = {}
        self._raw_to_entity: dict[tuple[str, int], int] = {}
        self._bag_anchors: dict[int, _BagAnchor] = {}

    def reset(self) -> None:
        self._next_entity_id = 1
        self._records.clear()
        self._raw_to_entity.clear()
        self._bag_anchors.clear()

    def update(
        self,
        tracks: list[TrackedObject],
        frame_idx: int,
        timestamp_ms: float,
        frame_shape: tuple[int, ...],
    ) -> list[EntityObservation]:
        self._purge_expired(timestamp_ms)
        observations: list[EntityObservation] = []
        observed_entity_ids: set[int] = set()

        grouped: dict[str, list[TrackedObject]] = {}
        for track in tracks:
            grouped.setdefault(track.label, []).append(track)

        ordered_labels = [self.labels.person, self.labels.bag]
        ordered_labels.extend(sorted(set(grouped) - set(ordered_labels)))

        for label in ordered_labels:
            label_tracks = grouped.get(label, [])
            if not label_tracks:
                continue
            label_observations = self._map_label_tracks(
                label_tracks,
                timestamp_ms,
                frame_shape,
                observed_entity_ids,
                observations,
            )
            observations.extend(label_observations)
            observed_entity_ids.update(item.entity_id for item in label_observations)

        # Emit short-lived inferred observations so relation/event state does not
        # collapse on a single missed detection.
        for record in self._records.values():
            if record.entity_id in observed_entity_ids:
                continue
            ttl = self._ttl_seconds(record.label)
            age_seconds = (timestamp_ms - record.last_seen_ms) / 1000.0
            if age_seconds < 0.0 or age_seconds > ttl:
                continue
            predicted_bbox = self._predict_bbox(record, timestamp_ms, observations)
            observations.append(
                EntityObservation(
                    entity_id=record.entity_id,
                    raw_track_id=None,
                    label=record.label,
                    class_id=record.class_id,
                    bbox=predicted_bbox,
                    confidence=0.0,
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    observed=False,
                    reliable=False,
                    identity_confidence=max(0.0, 1.0 - age_seconds / max(ttl, 1e-6)),
                )
            )
            record.last_update_ms = timestamp_ms
            record.last_frame_idx = frame_idx

        return sorted(observations, key=lambda item: (item.label, item.entity_id))

    def update_bag_anchors(
        self,
        relations: list[BagPersonRelation],
        observations: list[EntityObservation],
    ) -> None:
        """Remember bag position relative to its holder for the next frame."""
        by_id = {item.entity_id: item for item in observations}
        active_bag_ids: set[int] = set()
        for relation in relations:
            if relation.holder_person_id is None:
                continue
            bag = by_id.get(relation.bag_id)
            holder = by_id.get(relation.holder_person_id)
            if bag is None or holder is None or not bag.observed or not holder.observed:
                continue
            self._bag_anchors[relation.bag_id] = _BagAnchor(
                holder_person_id=holder.entity_id,
                relative_offset=relative_bbox_offset(bag.bbox, holder.bbox),
            )
            active_bag_ids.add(relation.bag_id)

        self._bag_anchors = {
            bag_id: anchor
            for bag_id, anchor in self._bag_anchors.items()
            if bag_id in self._records
        }

    def _map_label_tracks(
        self,
        tracks: list[TrackedObject],
        timestamp_ms: float,
        frame_shape: tuple[int, ...],
        already_observed: set[int],
        current_observations: list[EntityObservation],
    ) -> list[EntityObservation]:
        mapped: list[tuple[TrackedObject, int, float]] = []
        unknown: list[TrackedObject] = []
        used_entity_ids = set(already_observed)

        for track in tracks:
            entity_id = self._raw_to_entity.get((track.label, track.track_id))
            if entity_id is None or entity_id in used_entity_ids:
                unknown.append(track)
                continue
            mapped.append((track, entity_id, 1.0))
            used_entity_ids.add(entity_id)

        candidates = [
            record
            for record in self._records.values()
            if record.label == tracks[0].label
            and record.entity_id not in used_entity_ids
            and (timestamp_ms - record.last_seen_ms) / 1000.0 <= self._ttl_seconds(record.label)
        ]

        costs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(unknown):
            for record in candidates:
                predicted = self._predict_bbox(
                    record, timestamp_ms, current_observations
                )
                cost = self._reattach_cost(track.bbox, predicted, frame_shape)
                if cost <= self.config.reattach_max_cost:
                    costs.append((cost, track_index, record.entity_id))

        matched_tracks: set[int] = set()
        matched_entities: set[int] = set()
        for cost, track_index, entity_id in sorted(costs):
            if track_index in matched_tracks or entity_id in matched_entities:
                continue
            track = unknown[track_index]
            confidence = max(0.5, 1.0 - cost / self.config.reattach_max_cost)
            mapped.append((track, entity_id, confidence))
            matched_tracks.add(track_index)
            matched_entities.add(entity_id)
            used_entity_ids.add(entity_id)

        for track_index, track in enumerate(unknown):
            if track_index in matched_tracks:
                continue
            entity_id = self._create_record(track, timestamp_ms)
            mapped.append((track, entity_id, self.config.new_identity_confidence))
            used_entity_ids.add(entity_id)

        observations: list[EntityObservation] = []
        for track, entity_id, identity_confidence in mapped:
            self._update_record(entity_id, track, timestamp_ms)
            self._raw_to_entity[(track.label, track.track_id)] = entity_id
            observations.append(
                EntityObservation(
                    entity_id=entity_id,
                    raw_track_id=track.track_id,
                    label=track.label,
                    class_id=track.class_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    frame_idx=track.frame_idx,
                    timestamp_ms=track.timestamp_ms,
                    observed=True,
                    reliable=track.is_reliable,
                    identity_confidence=identity_confidence,
                )
            )
        return observations

    def _create_record(self, track: TrackedObject, timestamp_ms: float) -> int:
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self._records[entity_id] = _EntityRecord(
            entity_id=entity_id,
            label=track.label,
            class_id=track.class_id,
            last_bbox=track.bbox,
            last_seen_ms=timestamp_ms,
            last_update_ms=timestamp_ms,
            last_frame_idx=track.frame_idx,
            confidence=track.confidence,
            reliable=track.is_reliable,
            raw_track_ids={track.track_id},
        )
        return entity_id

    def _update_record(self, entity_id: int, track: TrackedObject, timestamp_ms: float) -> None:
        record = self._records[entity_id]
        dt = (timestamp_ms - record.last_seen_ms) / 1000.0
        if dt > 1e-3:
            previous_cx, previous_cy = bbox_center(record.last_bbox)
            current_cx, current_cy = bbox_center(track.bbox)
            record.velocity_x = (current_cx - previous_cx) / dt
            record.velocity_y = (current_cy - previous_cy) / dt
        record.last_bbox = track.bbox
        record.last_seen_ms = timestamp_ms
        record.last_update_ms = timestamp_ms
        record.last_frame_idx = track.frame_idx
        record.confidence = track.confidence
        record.reliable = track.is_reliable
        record.raw_track_ids.add(track.track_id)

    def _predict_bbox(
        self,
        record: _EntityRecord,
        timestamp_ms: float,
        current_observations: list[EntityObservation],
    ) -> BBox:
        if record.label == self.labels.bag:
            anchor = self._bag_anchors.get(record.entity_id)
            if anchor is not None:
                holder = next(
                    (
                        item
                        for item in current_observations
                        if item.entity_id == anchor.holder_person_id
                    ),
                    None,
                )
                if holder is not None:
                    return bbox_from_relative_offset(anchor.relative_offset, holder.bbox)

        dt = max(0.0, (timestamp_ms - record.last_seen_ms) / 1000.0)
        return translate_bbox(
            record.last_bbox,
            record.velocity_x * dt,
            record.velocity_y * dt,
        )

    def _reattach_cost(
        self,
        detected_bbox: BBox,
        predicted_bbox: BBox,
        frame_shape: tuple[int, ...],
    ) -> float:
        height, width = frame_shape[:2]
        frame_diagonal = max(math.hypot(width, height), 1.0)
        position_cost = math.dist(
            bbox_center(detected_bbox), bbox_center(predicted_bbox)
        ) / frame_diagonal
        detected_width, detected_height = bbox_size(detected_bbox)
        predicted_width, predicted_height = bbox_size(predicted_bbox)
        scale_cost = abs(
            math.log((detected_width * detected_height) / (predicted_width * predicted_height))
        )
        return (
            self.config.position_weight * position_cost
            + self.config.scale_weight * min(scale_cost, 1.0)
        )

    def _purge_expired(self, timestamp_ms: float) -> None:
        expired_ids = {
            entity_id
            for entity_id, record in self._records.items()
            if (timestamp_ms - record.last_seen_ms) / 1000.0 > self._ttl_seconds(record.label)
        }
        for entity_id in expired_ids:
            self._records.pop(entity_id, None)
            self._bag_anchors.pop(entity_id, None)
        self._raw_to_entity = {
            key: entity_id
            for key, entity_id in self._raw_to_entity.items()
            if entity_id not in expired_ids
        }

    def _ttl_seconds(self, label: str) -> float:
        if label == self.labels.person:
            return self.config.person_lost_ttl_seconds
        if label == self.labels.bag:
            return self.config.bag_lost_ttl_seconds
        return self.config.other_lost_ttl_seconds
