"""Liên kết người giữ túi theo thời gian với cơ chế hysteresis."""

from dataclasses import dataclass, field
import math

from src.core.analytics.config import AssociationConfig, LabelsConfig
from src.core.analytics.types import (
    BagPersonRelation,
    EntityObservation,
    MotionState,
    RelationState,
)
from src.core.analytics.geometry import (
    bbox_center,
    bbox_size,
    cosine_similarity,
    expand_bbox,
    point_to_bbox_distance,
    relative_bbox_offset,
)


@dataclass
class _RelationRecord:
    bag_id: int
    holder_person_id: int | None = None
    baseline_holder_id: int | None = None
    previous_holder_id: int | None = None
    candidate_holder_id: int | None = None
    state: RelationState = RelationState.UNASSOCIATED
    confidence: float = 0.0
    holder_since_ms: float | None = None
    candidate_since_ms: float | None = None
    detach_since_ms: float | None = None
    last_switch_ms: float | None = None
    previous_holder_duration_seconds: float = 0.0
    last_relative_offsets: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


class PersonBagAssociationManager:
    """Assign each bag to at most one person while tolerating short gaps."""

    def __init__(self, config: AssociationConfig, labels: LabelsConfig):
        self.config = config
        self.labels = labels
        self._records: dict[int, _RelationRecord] = {}

    def reset(self) -> None:
        self._records.clear()

    def update(
        self,
        observations: list[EntityObservation],
        motions: dict[int, MotionState],
        timestamp_ms: float,
    ) -> list[BagPersonRelation]:
        people = [item for item in observations if item.label == self.labels.person]
        bags = [item for item in observations if item.label == self.labels.bag]
        active_bag_ids = {bag.entity_id for bag in bags}

        relations: list[BagPersonRelation] = []
        for bag in bags:
            record = self._records.setdefault(
                bag.entity_id, _RelationRecord(bag_id=bag.entity_id)
            )
            scores = self._candidate_scores(record, bag, people, motions)
            self._advance_record(record, bag, people, scores, timestamp_ms)
            relations.append(self._snapshot(record, scores, timestamp_ms))

        # The identity manager owns the lifetime of a bag. Once it stops
        # emitting the entity, its relation cannot affect future events.
        self._records = {
            bag_id: record
            for bag_id, record in self._records.items()
            if bag_id in active_bag_ids
        }
        return sorted(relations, key=lambda item: item.bag_id)

    def _candidate_scores(
        self,
        record: _RelationRecord,
        bag: EntityObservation,
        people: list[EntityObservation],
        motions: dict[int, MotionState],
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        bag_center = bbox_center(bag.bbox)
        bag_motion = motions.get(bag.entity_id)

        for person in people:
            expanded = expand_bbox(person.bbox, self.config.expanded_person_box_ratio)
            _, person_height = bbox_size(person.bbox)
            distance = point_to_bbox_distance(bag_center, expanded) / person_height
            if distance > self.config.max_normalized_distance:
                continue

            spatial_score = max(
                0.0, 1.0 - distance / max(self.config.max_normalized_distance, 1e-6)
            )
            previous_holder_score = (
                1.0 if person.entity_id == record.holder_person_id else 0.0
            )
            person_motion = motions.get(person.entity_id)
            motion_score = self._motion_score(bag_motion, person_motion)
            relative_position_score = self._relative_position_score(record, bag, person)
            quality_score = self._quality_score(bag, person, bag_motion, person_motion)
            weights = self.config.weights
            score = (
                weights.spatial * spatial_score
                + weights.previous_holder * previous_holder_score
                + weights.motion * motion_score
                + weights.relative_position * relative_position_score
                + weights.quality * quality_score
            )
            scores[person.entity_id] = max(0.0, min(1.0, score))

        return scores

    def _advance_record(
        self,
        record: _RelationRecord,
        bag: EntityObservation,
        people: list[EntityObservation],
        scores: dict[int, float],
        timestamp_ms: float,
    ) -> None:
        people_by_id = {person.entity_id: person for person in people}
        current_score = scores.get(record.holder_person_id, 0.0)
        best_person_id, best_score = self._best_candidate(scores)

        if not bag.observed:
            if record.holder_person_id is not None:
                record.state = RelationState.LOST_ATTACHED
                record.confidence *= 0.92
            return

        if record.holder_person_id is None:
            self._advance_without_holder(record, best_person_id, best_score, timestamp_ms)
        else:
            should_switch = (
                best_person_id is not None
                and best_person_id != record.holder_person_id
                and best_score >= self.config.min_attach_score
                and best_score >= current_score + self.config.switch_margin
            )
            if should_switch:
                self._advance_switch(record, best_person_id, best_score, timestamp_ms)
            elif current_score >= self.config.min_hold_score:
                record.state = RelationState.ATTACHED
                record.confidence = current_score
                record.candidate_holder_id = None
                record.candidate_since_ms = None
                record.detach_since_ms = None
            else:
                self._advance_detach(record, timestamp_ms)

        holder = people_by_id.get(record.holder_person_id)
        if holder is not None and record.state == RelationState.ATTACHED:
            record.last_relative_offsets[holder.entity_id] = relative_bbox_offset(
                bag.bbox, holder.bbox
            )

        self._advance_baseline(record, timestamp_ms)

    def _advance_without_holder(
        self,
        record: _RelationRecord,
        best_person_id: int | None,
        best_score: float,
        timestamp_ms: float,
    ) -> None:
        if best_person_id is None or best_score < self.config.min_attach_score:
            record.state = RelationState.UNASSOCIATED
            record.confidence = best_score
            record.candidate_holder_id = None
            record.candidate_since_ms = None
            return

        if record.candidate_holder_id != best_person_id:
            record.candidate_holder_id = best_person_id
            record.candidate_since_ms = timestamp_ms
        record.state = RelationState.SWITCH_PENDING
        record.confidence = best_score

        candidate_age = self._elapsed_seconds(timestamp_ms, record.candidate_since_ms)
        if candidate_age >= self.config.attach_confirm_seconds:
            record.holder_person_id = best_person_id
            record.holder_since_ms = record.candidate_since_ms
            record.state = RelationState.ATTACHED
            record.candidate_holder_id = None
            record.candidate_since_ms = None
            record.detach_since_ms = None

    def _advance_switch(
        self,
        record: _RelationRecord,
        best_person_id: int,
        best_score: float,
        timestamp_ms: float,
    ) -> None:
        if record.candidate_holder_id != best_person_id:
            record.candidate_holder_id = best_person_id
            record.candidate_since_ms = timestamp_ms
        record.state = RelationState.SWITCH_PENDING
        record.confidence = best_score

        candidate_age = self._elapsed_seconds(timestamp_ms, record.candidate_since_ms)
        if candidate_age < self.config.switch_confirm_seconds:
            return

        previous_holder = record.holder_person_id
        previous_duration = self._elapsed_seconds(timestamp_ms, record.holder_since_ms)
        if record.baseline_holder_id is None and (
            previous_duration >= self.config.stable_holder_seconds
        ):
            record.baseline_holder_id = previous_holder

        record.previous_holder_id = previous_holder
        record.previous_holder_duration_seconds = previous_duration
        record.holder_person_id = best_person_id
        record.holder_since_ms = record.candidate_since_ms
        record.last_switch_ms = timestamp_ms
        record.state = RelationState.ATTACHED
        record.candidate_holder_id = None
        record.candidate_since_ms = None
        record.detach_since_ms = None

    def _advance_detach(self, record: _RelationRecord, timestamp_ms: float) -> None:
        if record.detach_since_ms is None:
            record.detach_since_ms = timestamp_ms
        record.state = RelationState.LOST_ATTACHED
        record.confidence *= 0.9
        if (
            self._elapsed_seconds(timestamp_ms, record.detach_since_ms)
            < self.config.detach_confirm_seconds
        ):
            return

        record.previous_holder_id = record.holder_person_id
        record.previous_holder_duration_seconds = self._elapsed_seconds(
            timestamp_ms, record.holder_since_ms
        )
        record.holder_person_id = None
        record.holder_since_ms = None
        record.state = RelationState.DETACHED

    def _advance_baseline(self, record: _RelationRecord, timestamp_ms: float) -> None:
        if record.holder_person_id is None:
            return
        holder_duration = self._elapsed_seconds(timestamp_ms, record.holder_since_ms)
        if (
            record.baseline_holder_id is None
            and holder_duration >= self.config.stable_holder_seconds
        ):
            record.baseline_holder_id = record.holder_person_id
            return
        if record.baseline_holder_id == record.holder_person_id:
            return
        if record.last_switch_ms is None:
            return
        if (
            self._elapsed_seconds(timestamp_ms, record.last_switch_ms)
            >= self.config.baseline_memory_seconds
        ):
            record.baseline_holder_id = record.holder_person_id
            record.previous_holder_id = None
            record.previous_holder_duration_seconds = 0.0

    def _snapshot(
        self,
        record: _RelationRecord,
        scores: dict[int, float],
        timestamp_ms: float,
    ) -> BagPersonRelation:
        holder_duration = self._elapsed_seconds(timestamp_ms, record.holder_since_ms)
        return BagPersonRelation(
            bag_id=record.bag_id,
            holder_person_id=record.holder_person_id,
            baseline_holder_id=record.baseline_holder_id,
            previous_holder_id=record.previous_holder_id,
            candidate_holder_id=record.candidate_holder_id,
            state=record.state,
            confidence=max(0.0, min(1.0, record.confidence)),
            holder_duration_seconds=holder_duration,
            previous_holder_duration_seconds=record.previous_holder_duration_seconds,
            candidate_scores=dict(scores),
            evidence={
                "holder_score": scores.get(record.holder_person_id, 0.0),
                "candidate_score": scores.get(record.candidate_holder_id, 0.0),
            },
        )

    def _relative_position_score(
        self,
        record: _RelationRecord,
        bag: EntityObservation,
        person: EntityObservation,
    ) -> float:
        previous = record.last_relative_offsets.get(person.entity_id)
        if previous is None:
            return 0.5
        current = relative_bbox_offset(bag.bbox, person.bbox)
        offset_distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
        return max(0.0, 1.0 - min(offset_distance, 1.0))

    @staticmethod
    def _motion_score(
        bag_motion: MotionState | None,
        person_motion: MotionState | None,
    ) -> float:
        if bag_motion is None or person_motion is None:
            return 0.5
        if bag_motion.speed < 0.005 or person_motion.speed < 0.005:
            return 0.5
        similarity = cosine_similarity(
            (bag_motion.vx, bag_motion.vy),
            (person_motion.vx, person_motion.vy),
        )
        return (similarity + 1.0) / 2.0

    @staticmethod
    def _quality_score(
        bag: EntityObservation,
        person: EntityObservation,
        bag_motion: MotionState | None,
        person_motion: MotionState | None,
    ) -> float:
        observation_quality = (
            float(bag.observed)
            + float(person.observed)
            + float(bag.reliable)
            + float(person.reliable)
        ) / 4.0
        motion_quality = (
            (bag_motion.quality if bag_motion else 0.0)
            + (person_motion.quality if person_motion else 0.0)
        ) / 2.0
        return (observation_quality + motion_quality) / 2.0

    @staticmethod
    def _best_candidate(scores: dict[int, float]) -> tuple[int | None, float]:
        if not scores:
            return None, 0.0
        person_id = max(scores, key=lambda candidate: (scores[candidate], -candidate))
        return person_id, scores[person_id]

    @staticmethod
    def _elapsed_seconds(timestamp_ms: float, start_ms: float | None) -> float:
        if start_ms is None:
            return 0.0
        return max(0.0, (timestamp_ms - start_ms) / 1000.0)
