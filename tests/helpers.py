"""Các factory test nhỏ cho những kịch bản analytics xác định."""

from src.core.analytics.types import (
    BagPersonRelation,
    EntityObservation,
    MotionState,
    RelationState,
)
from src.core.types import TrackedObject


def tracked(
    track_id: int,
    label: str,
    bbox: tuple[float, float, float, float],
    timestamp_ms: float,
    frame_idx: int = 0,
    confidence: float = 0.9,
    reliable: bool = True,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        class_id=0 if label == "person" else 1,
        label=label,
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        is_reliable=reliable,
    )


def entity(
    entity_id: int,
    label: str,
    bbox: tuple[float, float, float, float],
    timestamp_ms: float,
    observed: bool = True,
) -> EntityObservation:
    return EntityObservation(
        entity_id=entity_id,
        raw_track_id=entity_id,
        label=label,
        class_id=0 if label == "person" else 1,
        bbox=bbox,
        confidence=0.9 if observed else 0.0,
        frame_idx=int(timestamp_ms / 100.0),
        timestamp_ms=timestamp_ms,
        observed=observed,
        reliable=observed,
        identity_confidence=1.0,
    )


def motion(
    entity_id: int,
    vx: float = 0.0,
    vy: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    quality: float = 1.0,
) -> MotionState:
    return MotionState(
        entity_id=entity_id,
        x=0.0,
        y=0.0,
        scale=0.0,
        vx=vx,
        vy=vy,
        scale_velocity=0.0,
        ax=ax,
        ay=ay,
        speed=(vx * vx + vy * vy) ** 0.5,
        quality=quality,
    )


def relation(
    holder: int | None,
    *,
    bag_id: int = 3,
    baseline: int | None = 1,
    previous: int | None = None,
    candidate: int | None = None,
    holder_duration: float = 2.0,
    previous_duration: float = 0.0,
) -> BagPersonRelation:
    return BagPersonRelation(
        bag_id=bag_id,
        holder_person_id=holder,
        baseline_holder_id=baseline,
        previous_holder_id=previous,
        candidate_holder_id=candidate,
        state=(RelationState.SWITCH_PENDING if candidate else RelationState.ATTACHED),
        confidence=0.9,
        holder_duration_seconds=holder_duration,
        previous_holder_duration_seconds=previous_duration,
    )
