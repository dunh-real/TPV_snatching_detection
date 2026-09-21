"""Facade thực thi toàn bộ pipeline analytics dựa trên luật."""

from pathlib import Path

from src.core.analytics.config import RuleEngineConfig, load_rule_engine_config
from src.core.analytics.identity import EntityIdentityManager
from src.core.analytics.types import AnalyticsResult
from src.core.analytics.person_bag import PersonBagAssociationManager
from src.core.analytics.roles import RoleResolver
from src.core.analytics.snatch_detector import SnatchDetector
from src.core.analytics.trajectory import TrajectoryManager
from src.core.types import TrackedObject


class SnatchAnalyticsEngine:
    """Stable public interface for stateful person-bag analytics."""

    def __init__(self, config: RuleEngineConfig):
        self.config = config
        self.identity = EntityIdentityManager(config.identity, config.labels)
        self.trajectory = TrajectoryManager(config.motion)
        self.person_bag = PersonBagAssociationManager(config.association, config.labels)
        self.snatch = SnatchDetector(
            config.snatch,
            terminal_retention_seconds=config.roles.display_role_seconds,
        )
        self.roles = RoleResolver(config.roles)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "SnatchAnalyticsEngine":
        return cls(load_rule_engine_config(path))

    def reset(self) -> None:
        self.identity.reset()
        self.trajectory.reset()
        self.person_bag.reset()
        self.snatch.reset()
        self.roles.reset()

    def update(
        self,
        tracked_objects: list[TrackedObject],
        frame_idx: int,
        timestamp_ms: float,
        frame_shape: tuple[int, ...],
    ) -> AnalyticsResult:
        entities = self.identity.update(
            tracked_objects,
            frame_idx,
            timestamp_ms,
            frame_shape,
        )
        motions = self.trajectory.update(entities, frame_shape)
        relations = self.person_bag.update(entities, motions, timestamp_ms)
        self.identity.update_bag_anchors(relations, entities)
        events = self.snatch.update(
            entities,
            motions,
            relations,
            timestamp_ms,
            self.config.labels.person,
        )
        roles = self.roles.update(entities, events, self.config.labels.person, timestamp_ms)
        return AnalyticsResult(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            entities=entities,
            motions=motions,
            relations=relations,
            events=events,
            roles=roles,
        )
