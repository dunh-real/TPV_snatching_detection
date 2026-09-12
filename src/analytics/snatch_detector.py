"""Explainable temporal rules for suspected bag-snatching events."""

from dataclasses import dataclass, field
import math

from src.analytics.config import SnatchConfig
from src.analytics.geometry import (
    bbox_center,
    cosine_similarity,
    normalized_center_distance,
    normalized_person_distance,
    normalized_point_to_person_distance,
)
from src.analytics.models import (
    BagPersonRelation,
    EntityObservation,
    EventState,
    MotionState,
    SnatchEvent,
)


@dataclass
class _Hypothesis:
    event_id: int
    bag_id: int
    victim_person_id: int
    suspect_person_id: int
    state: EventState
    start_ms: float
    state_since_ms: float
    updated_ms: float
    end_ms: float | None = None
    evidence: dict[str, float | bool] = field(default_factory=dict)
    last_timestamp_ms: float | None = None
    last_bag_suspect_distance: float | None = None
    last_bag_victim_distance: float | None = None
    last_person_separation: float | None = None
    approach_since_ms: float | None = None
    contact_since_ms: float | None = None
    escape_since_ms: float | None = None


class SnatchDetector:
    """Maintain one stateful hypothesis per bag-victim-suspect triplet."""

    def __init__(
        self,
        config: SnatchConfig,
        terminal_retention_seconds: float | None = None,
    ):
        self.config = config
        self.terminal_retention_seconds = (
            terminal_retention_seconds
            if terminal_retention_seconds is not None
            else config.event_timeout_seconds
        )
        self._next_event_id = 1
        self._hypotheses: dict[tuple[int, int, int], _Hypothesis] = {}

    def reset(self) -> None:
        self._next_event_id = 1
        self._hypotheses.clear()

    def update(
        self,
        observations: list[EntityObservation],
        motions: dict[int, MotionState],
        relations: list[BagPersonRelation],
        timestamp_ms: float,
        person_label: str,
    ) -> list[SnatchEvent]:
        by_id = {item.entity_id: item for item in observations}
        people = [item for item in observations if item.label == person_label]
        relations_by_bag = {item.bag_id: item for item in relations}

        self._create_approach_hypotheses(people, by_id, relations, timestamp_ms)
        self._create_direct_transfer_hypotheses(relations, timestamp_ms)

        for hypothesis in list(self._hypotheses.values()):
            if hypothesis.state in {EventState.SUSPECTED, EventState.CANCELLED}:
                continue
            self._advance(
                hypothesis,
                by_id,
                motions,
                relations_by_bag.get(hypothesis.bag_id),
                timestamp_ms,
            )

        snapshots = [
            self._snapshot(item)
            for item in self._hypotheses.values()
            if item.state != EventState.APPROACHING or item.evidence.get("approach")
        ]
        self._purge_terminal(timestamp_ms)
        return sorted(snapshots, key=lambda item: item.event_id)

    def _create_approach_hypotheses(
        self,
        people: list[EntityObservation],
        by_id: dict[int, EntityObservation],
        relations: list[BagPersonRelation],
        timestamp_ms: float,
    ) -> None:
        for relation in relations:
            victim_id = relation.holder_person_id
            if (
                victim_id is None
                or relation.holder_duration_seconds < self.config.min_owner_duration_seconds
            ):
                continue
            bag = by_id.get(relation.bag_id)
            if bag is None or not bag.observed:
                continue
            for suspect in people:
                if suspect.entity_id == victim_id or not suspect.observed:
                    continue
                distance = normalized_center_distance(bag.bbox, suspect.bbox)
                if distance > self.config.approach_max_distance:
                    continue
                key = (bag.entity_id, victim_id, suspect.entity_id)
                if key not in self._hypotheses:
                    self._hypotheses[key] = self._new_hypothesis(
                        bag.entity_id,
                        victim_id,
                        suspect.entity_id,
                        EventState.APPROACHING,
                        timestamp_ms,
                    )

    def _create_direct_transfer_hypotheses(
        self,
        relations: list[BagPersonRelation],
        timestamp_ms: float,
    ) -> None:
        for relation in relations:
            victim_id = relation.previous_holder_id or relation.baseline_holder_id
            suspect_id = relation.holder_person_id
            if (
                victim_id is None
                or suspect_id is None
                or victim_id == suspect_id
                or relation.previous_holder_duration_seconds
                < self.config.min_owner_duration_seconds
            ):
                continue
            key = (relation.bag_id, victim_id, suspect_id)
            hypothesis = self._hypotheses.get(key)
            if hypothesis is None:
                hypothesis = self._new_hypothesis(
                    relation.bag_id,
                    victim_id,
                    suspect_id,
                    EventState.ESCAPING,
                    timestamp_ms,
                )
                self._hypotheses[key] = hypothesis
            hypothesis.evidence["stable_holder"] = True
            hypothesis.evidence["new_holder"] = True
            hypothesis.evidence["holder_switch"] = True
            if hypothesis.state not in {EventState.SUSPECTED, EventState.CANCELLED}:
                self._set_state(hypothesis, EventState.ESCAPING, timestamp_ms)

    def _advance(
        self,
        hypothesis: _Hypothesis,
        by_id: dict[int, EntityObservation],
        motions: dict[int, MotionState],
        relation: BagPersonRelation | None,
        timestamp_ms: float,
    ) -> None:
        bag = by_id.get(hypothesis.bag_id)
        victim = by_id.get(hypothesis.victim_person_id)
        suspect = by_id.get(hypothesis.suspect_person_id)
        if bag is None or victim is None or suspect is None or relation is None:
            if (
                self._elapsed(timestamp_ms, hypothesis.updated_ms)
                > self.config.event_timeout_seconds
            ):
                self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)
            return

        dt = max(
            self._elapsed(timestamp_ms, hypothesis.last_timestamp_ms),
            1e-3,
        )
        bag_suspect_distance = normalized_center_distance(bag.bbox, suspect.bbox)
        bag_victim_distance = normalized_center_distance(bag.bbox, victim.bbox)
        person_separation = normalized_person_distance(victim.bbox, suspect.bbox)
        contact_distance = normalized_point_to_person_distance(
            bbox_center(bag.bbox), suspect.bbox
        )

        approach_rate = self._decrease_rate(
            hypothesis.last_bag_suspect_distance, bag_suspect_distance, dt
        )
        bag_victim_separation_rate = self._increase_rate(
            hypothesis.last_bag_victim_distance, bag_victim_distance, dt
        )
        person_separation_rate = self._increase_rate(
            hypothesis.last_person_separation, person_separation, dt
        )

        self._update_observation_evidence(hypothesis, bag, victim, suspect, motions)
        self._update_interaction_evidence(
            hypothesis,
            relation,
            approach_rate,
            contact_distance,
            bag_victim_distance,
            bag_victim_separation_rate,
            timestamp_ms,
        )
        self._update_escape_evidence(
            hypothesis,
            relation,
            victim,
            suspect,
            motions,
            person_separation,
            person_separation_rate,
            timestamp_ms,
        )
        self._advance_state(hypothesis, relation, timestamp_ms)

        hypothesis.last_timestamp_ms = timestamp_ms
        hypothesis.last_bag_suspect_distance = bag_suspect_distance
        hypothesis.last_bag_victim_distance = bag_victim_distance
        hypothesis.last_person_separation = person_separation
        hypothesis.updated_ms = timestamp_ms

    def _update_interaction_evidence(
        self,
        hypothesis: _Hypothesis,
        relation: BagPersonRelation,
        approach_rate: float,
        contact_distance: float,
        bag_victim_distance: float,
        bag_victim_separation_rate: float,
        timestamp_ms: float,
    ) -> None:
        if approach_rate >= self.config.approach_min_rate:
            if hypothesis.approach_since_ms is None:
                hypothesis.approach_since_ms = timestamp_ms
            if (
                self._elapsed(timestamp_ms, hypothesis.approach_since_ms)
                >= self.config.approach_min_seconds
            ):
                hypothesis.evidence["approach"] = True
        elif not hypothesis.evidence.get("approach"):
            hypothesis.approach_since_ms = None

        contact = contact_distance <= self.config.contact_distance
        if contact:
            if hypothesis.contact_since_ms is None:
                hypothesis.contact_since_ms = timestamp_ms
            hypothesis.evidence["contact"] = True
            if hypothesis.state == EventState.APPROACHING:
                self._set_state(hypothesis, EventState.CONTACT, timestamp_ms)

        if (
            not hypothesis.evidence.get("contact")
            and relation.candidate_holder_id == hypothesis.suspect_person_id
        ):
            hypothesis.evidence["contact"] = True

        if (
            bag_victim_distance >= self.config.escape_min_separation / 2.0
            and bag_victim_separation_rate >= self.config.escape_min_rate
        ):
            hypothesis.evidence["bag_decoupled"] = True

        if relation.candidate_holder_id == hypothesis.suspect_person_id:
            if hypothesis.state in {EventState.APPROACHING, EventState.CONTACT}:
                self._set_state(hypothesis, EventState.TRANSFER_PENDING, timestamp_ms)

        switched = (
            relation.holder_person_id == hypothesis.suspect_person_id
            and (
                relation.previous_holder_id == hypothesis.victim_person_id
                or relation.baseline_holder_id == hypothesis.victim_person_id
            )
        )
        if switched:
            hypothesis.evidence["new_holder"] = True
            hypothesis.evidence["holder_switch"] = True
            self._set_state(hypothesis, EventState.ESCAPING, timestamp_ms)

    def _update_escape_evidence(
        self,
        hypothesis: _Hypothesis,
        relation: BagPersonRelation,
        victim: EntityObservation,
        suspect: EntityObservation,
        motions: dict[int, MotionState],
        person_separation: float,
        person_separation_rate: float,
        timestamp_ms: float,
    ) -> None:
        if hypothesis.state != EventState.ESCAPING:
            return
        bag_motion = motions.get(hypothesis.bag_id)
        suspect_motion = motions.get(hypothesis.suspect_person_id)
        victim_motion = motions.get(hypothesis.victim_person_id)
        if bag_motion is None or suspect_motion is None:
            return

        motion_similarity = cosine_similarity(
            (bag_motion.vx, bag_motion.vy),
            (suspect_motion.vx, suspect_motion.vy),
        )
        escaping = (
            relation.holder_person_id == hypothesis.suspect_person_id
            and person_separation >= self.config.escape_min_separation
            and person_separation_rate >= self.config.escape_min_rate
            and suspect_motion.speed >= self.config.suspect_min_speed
            and motion_similarity >= self.config.bag_suspect_motion_similarity
        )
        if escaping:
            if hypothesis.escape_since_ms is None:
                hypothesis.escape_since_ms = timestamp_ms
            if (
                self._elapsed(timestamp_ms, hypothesis.escape_since_ms)
                >= self.config.minimum_escape_seconds
            ):
                hypothesis.evidence["escape"] = True
        elif not hypothesis.evidence.get("escape"):
            hypothesis.escape_since_ms = None

        suspect_acceleration = math.hypot(suspect_motion.ax, suspect_motion.ay)
        if suspect_acceleration >= self.config.suspect_acceleration_threshold:
            hypothesis.evidence["suspect_acceleration"] = True

        if victim_motion is not None and victim_motion.speed >= self.config.victim_reaction_speed:
            victim_center = bbox_center(victim.bbox)
            suspect_center = bbox_center(suspect.bbox)
            direction_to_suspect = (
                suspect_center[0] - victim_center[0],
                suspect_center[1] - victim_center[1],
            )
            reaction_similarity = cosine_similarity(
                (victim_motion.vx, victim_motion.vy), direction_to_suspect
            )
            if reaction_similarity > 0.3:
                hypothesis.evidence["victim_reaction"] = True

    def _update_observation_evidence(
        self,
        hypothesis: _Hypothesis,
        bag: EntityObservation,
        victim: EntityObservation,
        suspect: EntityObservation,
        motions: dict[int, MotionState],
    ) -> None:
        qualities = [
            motions[item.entity_id].quality
            for item in (bag, victim, suspect)
            if item.entity_id in motions
        ]
        quality = sum(qualities) / len(qualities) if qualities else 0.0
        hypothesis.evidence["observation_quality"] = quality
        hypothesis.evidence["low_quality"] = quality < self.config.min_evidence_quality
        if not bag.observed and hypothesis.evidence.get("contact"):
            hypothesis.evidence["bag_missing_during_contact"] = True

    def _advance_state(
        self,
        hypothesis: _Hypothesis,
        relation: BagPersonRelation,
        timestamp_ms: float,
    ) -> None:
        state_age = self._elapsed(timestamp_ms, hypothesis.state_since_ms)
        event_age = self._elapsed(timestamp_ms, hypothesis.start_ms)

        if relation.holder_person_id == hypothesis.victim_person_id and (
            hypothesis.state in {EventState.TRANSFER_PENDING, EventState.ESCAPING}
        ):
            if state_age >= self.config.transfer_window_seconds:
                self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)
                return

        if hypothesis.state == EventState.APPROACHING:
            if state_age > self.config.approach_timeout_seconds:
                self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)
                return
        elif hypothesis.state == EventState.CONTACT:
            if state_age > self.config.contact_timeout_seconds:
                self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)
                return
        elif hypothesis.state == EventState.TRANSFER_PENDING:
            if state_age > self.config.transfer_window_seconds:
                self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)
                return
        elif hypothesis.state == EventState.ESCAPING:
            score = self._score(hypothesis)
            core_evidence = (
                bool(hypothesis.evidence.get("stable_holder"))
                and bool(hypothesis.evidence.get("holder_switch"))
                and bool(hypothesis.evidence.get("escape"))
            )
            if core_evidence and score >= self.config.suspected_score_threshold:
                self._finish(hypothesis, EventState.SUSPECTED, timestamp_ms)
                return
            if state_age > self.config.escape_window_seconds:
                self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)
                return

        if event_age > self.config.event_timeout_seconds:
            self._finish(hypothesis, EventState.CANCELLED, timestamp_ms)

    def _new_hypothesis(
        self,
        bag_id: int,
        victim_id: int,
        suspect_id: int,
        state: EventState,
        timestamp_ms: float,
    ) -> _Hypothesis:
        hypothesis = _Hypothesis(
            event_id=self._next_event_id,
            bag_id=bag_id,
            victim_person_id=victim_id,
            suspect_person_id=suspect_id,
            state=state,
            start_ms=timestamp_ms,
            state_since_ms=timestamp_ms,
            updated_ms=timestamp_ms,
            evidence={"stable_holder": True},
        )
        self._next_event_id += 1
        return hypothesis

    def _set_state(
        self,
        hypothesis: _Hypothesis,
        state: EventState,
        timestamp_ms: float,
    ) -> None:
        if hypothesis.state == state:
            return
        hypothesis.state = state
        hypothesis.state_since_ms = timestamp_ms
        hypothesis.updated_ms = timestamp_ms

    def _finish(
        self,
        hypothesis: _Hypothesis,
        state: EventState,
        timestamp_ms: float,
    ) -> None:
        self._set_state(hypothesis, state, timestamp_ms)
        hypothesis.end_ms = timestamp_ms

    def _score(self, hypothesis: _Hypothesis) -> float:
        scores = self.config.scores
        score = 0.0
        for key in (
            "stable_holder",
            "approach",
            "contact",
            "bag_decoupled",
            "new_holder",
            "holder_switch",
            "escape",
            "suspect_acceleration",
            "victim_reaction",
        ):
            if hypothesis.evidence.get(key):
                score += float(getattr(scores, key))
        if hypothesis.evidence.get("low_quality"):
            score -= scores.low_quality_penalty
        return max(0.0, score)

    def _snapshot(self, hypothesis: _Hypothesis) -> SnatchEvent:
        score = self._score(hypothesis)
        confidence = min(1.0, score / max(self.config.suspected_score_threshold, 1e-6))
        return SnatchEvent(
            event_id=hypothesis.event_id,
            bag_id=hypothesis.bag_id,
            victim_person_id=hypothesis.victim_person_id,
            suspect_person_id=hypothesis.suspect_person_id,
            state=hypothesis.state,
            score=score,
            confidence=confidence,
            start_ms=hypothesis.start_ms,
            updated_ms=hypothesis.updated_ms,
            end_ms=hypothesis.end_ms,
            evidence=dict(hypothesis.evidence),
        )

    def _purge_terminal(self, timestamp_ms: float) -> None:
        self._hypotheses = {
            key: hypothesis
            for key, hypothesis in self._hypotheses.items()
            if hypothesis.end_ms is None
            or self._elapsed(timestamp_ms, hypothesis.end_ms)
            <= self.terminal_retention_seconds
        }

    @staticmethod
    def _decrease_rate(previous: float | None, current: float, dt: float) -> float:
        if previous is None:
            return 0.0
        return (previous - current) / dt

    @staticmethod
    def _increase_rate(previous: float | None, current: float, dt: float) -> float:
        if previous is None:
            return 0.0
        return (current - previous) / dt

    @staticmethod
    def _elapsed(timestamp_ms: float, start_ms: float | None) -> float:
        if start_ms is None:
            return 0.0
        return max(0.0, (timestamp_ms - start_ms) / 1000.0)
