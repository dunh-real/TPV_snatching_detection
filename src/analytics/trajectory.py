"""Smoothed online kinematics for canonical tracked entities."""

from dataclasses import dataclass
import math

from src.analytics.config import MotionConfig
from src.analytics.geometry import bbox_center, normalized_scale
from src.analytics.models import EntityObservation, MotionState


@dataclass
class _TrajectoryState:
    timestamp_ms: float
    x: float
    y: float
    scale: float
    vx: float = 0.0
    vy: float = 0.0
    scale_velocity: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    quality: float = 0.0


class TrajectoryManager:
    """Calculate stable normalized velocity and acceleration per entity."""

    def __init__(self, config: MotionConfig):
        self.config = config
        self._states: dict[int, _TrajectoryState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(
        self,
        observations: list[EntityObservation],
        frame_shape: tuple[int, ...],
    ) -> dict[int, MotionState]:
        height, width = frame_shape[:2]
        result: dict[int, MotionState] = {}

        for observation in observations:
            cx, cy = bbox_center(observation.bbox)
            measured_x = cx / max(width, 1)
            measured_y = cy / max(height, 1)
            measured_scale = normalized_scale(observation.bbox, frame_shape)
            previous = self._states.get(observation.entity_id)

            if previous is None:
                quality = self._measurement_quality(observation)
                current = _TrajectoryState(
                    timestamp_ms=observation.timestamp_ms,
                    x=measured_x,
                    y=measured_y,
                    scale=measured_scale,
                    quality=quality,
                )
            else:
                current = self._advance(
                    previous,
                    observation,
                    measured_x,
                    measured_y,
                    measured_scale,
                )

            self._states[observation.entity_id] = current
            result[observation.entity_id] = MotionState(
                entity_id=observation.entity_id,
                x=current.x,
                y=current.y,
                scale=current.scale,
                vx=current.vx,
                vy=current.vy,
                scale_velocity=current.scale_velocity,
                ax=current.ax,
                ay=current.ay,
                speed=math.hypot(current.vx, current.vy),
                quality=current.quality,
            )

        active_ids = {observation.entity_id for observation in observations}
        self._states = {
            entity_id: state
            for entity_id, state in self._states.items()
            if entity_id in active_ids
        }
        return result

    def _advance(
        self,
        previous: _TrajectoryState,
        observation: EntityObservation,
        measured_x: float,
        measured_y: float,
        measured_scale: float,
    ) -> _TrajectoryState:
        dt = max(
            (observation.timestamp_ms - previous.timestamp_ms) / 1000.0,
            self.config.min_dt_seconds,
        )

        if observation.observed:
            alpha = self.config.ema_alpha
            x = alpha * measured_x + (1.0 - alpha) * previous.x
            y = alpha * measured_y + (1.0 - alpha) * previous.y
            scale = alpha * measured_scale + (1.0 - alpha) * previous.scale
            raw_vx = (x - previous.x) / dt
            raw_vy = (y - previous.y) / dt
            raw_scale_velocity = (scale - previous.scale) / dt
            velocity_alpha = self.config.velocity_alpha
            vx = velocity_alpha * raw_vx + (1.0 - velocity_alpha) * previous.vx
            vy = velocity_alpha * raw_vy + (1.0 - velocity_alpha) * previous.vy
            scale_velocity = (
                velocity_alpha * raw_scale_velocity
                + (1.0 - velocity_alpha) * previous.scale_velocity
            )
            raw_ax = (vx - previous.vx) / dt
            raw_ay = (vy - previous.vy) / dt
            acceleration_alpha = self.config.acceleration_alpha
            ax = acceleration_alpha * raw_ax + (1.0 - acceleration_alpha) * previous.ax
            ay = acceleration_alpha * raw_ay + (1.0 - acceleration_alpha) * previous.ay
            quality = self._measurement_quality(observation)
        else:
            # IdentityManager already supplies a predicted box. Preserve prior
            # velocity and mark it as inferred instead of differentiating it again.
            x, y, scale = measured_x, measured_y, measured_scale
            vx, vy = previous.vx, previous.vy
            scale_velocity = previous.scale_velocity
            ax, ay = previous.ax, previous.ay
            quality = previous.quality * (
                self.config.inferred_quality_decay_per_second**dt
            )

        return _TrajectoryState(
            timestamp_ms=observation.timestamp_ms,
            x=x,
            y=y,
            scale=scale,
            vx=vx,
            vy=vy,
            scale_velocity=scale_velocity,
            ax=ax,
            ay=ay,
            quality=max(0.0, min(1.0, quality)),
        )

    @staticmethod
    def _measurement_quality(observation: EntityObservation) -> float:
        confidence_quality = max(0.0, min(1.0, observation.confidence))
        reliability_factor = 1.0 if observation.reliable else 0.55
        return confidence_quality * reliability_factor * observation.identity_confidence
