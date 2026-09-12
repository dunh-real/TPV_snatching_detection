"""Typed configuration loader for the person-bag rule engine."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class LabelsConfig:
    person: str = "person"
    bag: str = "bag"


@dataclass(frozen=True)
class MotionConfig:
    ema_alpha: float = 0.35
    velocity_alpha: float = 0.45
    acceleration_alpha: float = 0.35
    inferred_quality_decay_per_second: float = 0.75
    min_dt_seconds: float = 1e-3


@dataclass(frozen=True)
class IdentityConfig:
    person_lost_ttl_seconds: float = 2.0
    bag_lost_ttl_seconds: float = 2.0
    other_lost_ttl_seconds: float = 1.0
    reattach_max_cost: float = 0.22
    position_weight: float = 0.80
    scale_weight: float = 0.20
    new_identity_confidence: float = 0.55


@dataclass(frozen=True)
class AssociationWeights:
    spatial: float = 0.30
    previous_holder: float = 0.25
    motion: float = 0.20
    relative_position: float = 0.15
    quality: float = 0.10


@dataclass(frozen=True)
class AssociationConfig:
    expanded_person_box_ratio: float = 0.20
    max_normalized_distance: float = 0.75
    min_attach_score: float = 0.52
    min_hold_score: float = 0.38
    switch_margin: float = 0.08
    attach_confirm_seconds: float = 0.30
    stable_holder_seconds: float = 0.80
    switch_confirm_seconds: float = 0.30
    detach_confirm_seconds: float = 0.50
    baseline_memory_seconds: float = 5.0
    weights: AssociationWeights = field(default_factory=AssociationWeights)


@dataclass(frozen=True)
class SnatchScores:
    stable_holder: float = 1.0
    approach: float = 1.0
    contact: float = 1.0
    bag_decoupled: float = 2.0
    new_holder: float = 2.0
    holder_switch: float = 2.0
    escape: float = 2.0
    suspect_acceleration: float = 1.0
    victim_reaction: float = 1.0
    low_quality_penalty: float = 1.0


@dataclass(frozen=True)
class SnatchConfig:
    min_owner_duration_seconds: float = 0.80
    approach_max_distance: float = 2.0
    approach_min_rate: float = 0.08
    approach_min_seconds: float = 0.20
    approach_timeout_seconds: float = 1.50
    contact_distance: float = 0.15
    contact_timeout_seconds: float = 1.50
    transfer_window_seconds: float = 2.0
    escape_window_seconds: float = 3.0
    event_timeout_seconds: float = 6.0
    minimum_escape_seconds: float = 0.40
    escape_min_separation: float = 0.75
    escape_min_rate: float = 0.05
    bag_suspect_motion_similarity: float = 0.45
    suspect_min_speed: float = 0.03
    suspect_acceleration_threshold: float = 0.05
    victim_reaction_speed: float = 0.025
    min_evidence_quality: float = 0.35
    suspected_score_threshold: float = 7.0
    scores: SnatchScores = field(default_factory=SnatchScores)


@dataclass(frozen=True)
class RolesConfig:
    display_role_seconds: float = 5.0
    grace_period_seconds: float = 1.0
    confirmed_lock: bool = True


@dataclass(frozen=True)
class RuleEngineConfig:
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    snatch: SnatchConfig = field(default_factory=SnatchConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    version: str = "1.0"

    def validate(self) -> None:
        """Reject invalid values early instead of failing inside a video run."""
        if not 0.0 < self.motion.ema_alpha <= 1.0:
            raise ValueError("motion.ema_alpha must be in (0, 1]")
        if not 0.0 < self.motion.velocity_alpha <= 1.0:
            raise ValueError("motion.velocity_alpha must be in (0, 1]")
        if not 0.0 < self.motion.inferred_quality_decay_per_second <= 1.0:
            raise ValueError(
                "motion.inferred_quality_decay_per_second must be in (0, 1]"
            )
        if self.identity.reattach_max_cost <= 0.0:
            raise ValueError("identity.reattach_max_cost must be positive")
        if self.association.switch_confirm_seconds < 0.0:
            raise ValueError("association.switch_confirm_seconds cannot be negative")
        if self.snatch.suspected_score_threshold <= 0.0:
            raise ValueError("snatch.suspected_score_threshold must be positive")


T = TypeVar("T")


def _from_mapping(cls: type[T], values: dict[str, Any] | None, **nested: Any) -> T:
    values = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {', '.join(unknown)}")
    values.update(nested)
    return cls(**values)


def load_rule_engine_config(path: str | Path | None = None) -> RuleEngineConfig:
    """Load a YAML file on top of explicit dataclass defaults."""
    if path is None:
        config = RuleEngineConfig()
        config.validate()
        return config

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("Rule-engine config root must be a mapping")

    allowed = {item.name for item in fields(RuleEngineConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown RuleEngineConfig settings: {', '.join(unknown)}")

    association_raw = dict(raw.get("association") or {})
    weights = _from_mapping(AssociationWeights, association_raw.pop("weights", None))
    snatch_raw = dict(raw.get("snatch") or {})
    scores = _from_mapping(SnatchScores, snatch_raw.pop("scores", None))

    config = RuleEngineConfig(
        labels=_from_mapping(LabelsConfig, raw.get("labels")),
        motion=_from_mapping(MotionConfig, raw.get("motion")),
        identity=_from_mapping(IdentityConfig, raw.get("identity")),
        association=_from_mapping(AssociationConfig, association_raw, weights=weights),
        snatch=_from_mapping(SnatchConfig, snatch_raw, scores=scores),
        roles=_from_mapping(RolesConfig, raw.get("roles")),
        version=str(raw.get("version", "1.0")),
    )
    config.validate()
    return config
