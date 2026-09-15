"""Các kiểu dữ liệu nghiệp vụ cho analytics người-túi và cướp giật."""

from dataclasses import dataclass, field
from enum import Enum

from src.core.types import BBox


class RelationState(str, Enum):
    """Temporal state of a bag-person association."""

    UNASSOCIATED = "unassociated"
    ATTACHED = "attached"
    LOST_ATTACHED = "lost_attached"
    SWITCH_PENDING = "switch_pending"
    DETACHED = "detached"


class EventState(str, Enum):
    """Progress of one rule-based snatching hypothesis."""

    APPROACHING = "approaching"
    CONTACT = "contact"
    TRANSFER_PENDING = "transfer_pending"
    ESCAPING = "escaping"
    SUSPECTED = "suspected"
    CANCELLED = "cancelled"


class PersonRole(str, Enum):
    """Event-scoped role assigned to a tracked person."""

    NORMAL = "normal"
    POSSIBLE_VICTIM = "possible_victim"
    POSSIBLE_SUSPECT = "possible_suspect"
    VICTIM = "victim"
    SUSPECT = "suspect"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EntityObservation:
    """One raw tracker observation mapped onto a stable entity ID."""

    entity_id: int
    raw_track_id: int | None
    label: str
    class_id: int
    bbox: BBox
    confidence: float
    frame_idx: int
    timestamp_ms: float
    observed: bool = True
    reliable: bool = True
    identity_confidence: float = 1.0


@dataclass(frozen=True)
class MotionState:
    """Smoothed normalized kinematics for one entity."""

    entity_id: int
    x: float
    y: float
    scale: float
    vx: float
    vy: float
    scale_velocity: float
    ax: float
    ay: float
    speed: float
    quality: float


@dataclass(frozen=True)
class BagPersonRelation:
    """Current and recent holder information for one bag."""

    bag_id: int
    holder_person_id: int | None
    baseline_holder_id: int | None
    previous_holder_id: int | None
    candidate_holder_id: int | None
    state: RelationState
    confidence: float
    holder_duration_seconds: float
    previous_holder_duration_seconds: float
    candidate_scores: dict[int, float] = field(default_factory=dict)
    evidence: dict[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SnatchEvent:
    """Public snapshot of a snatching hypothesis."""

    event_id: int
    bag_id: int
    victim_person_id: int
    suspect_person_id: int
    state: EventState
    score: float
    confidence: float
    start_ms: float
    updated_ms: float
    end_ms: float | None = None
    evidence: dict[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleAssignment:
    """Role of a person in the strongest active event for this frame."""

    person_id: int
    role: PersonRole
    confidence: float
    event_id: int | None = None


@dataclass(frozen=True)
class AnalyticsResult:
    """All analytics emitted for one frame."""

    frame_idx: int
    timestamp_ms: float
    entities: list[EntityObservation]
    motions: dict[int, MotionState]
    relations: list[BagPersonRelation]
    events: list[SnatchEvent]
    roles: dict[int, RoleAssignment]
