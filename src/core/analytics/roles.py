"""Chuyển các giả thuyết theo sự kiện thành một vai trò hiển thị cho mỗi người."""

from dataclasses import dataclass

from src.core.analytics.config import RolesConfig
from src.core.analytics.types import (
    EntityObservation,
    EventState,
    PersonRole,
    RoleAssignment,
    SnatchEvent,
)


_STATE_PRIORITY = {
    EventState.APPROACHING: 1,
    EventState.CONTACT: 2,
    EventState.TRANSFER_PENDING: 3,
    EventState.ESCAPING: 4,
    EventState.SUSPECTED: 5,
    EventState.CANCELLED: 0,
}

# Confirmed roles cannot be downgraded.
_CONFIRMED_ROLES = {PersonRole.VICTIM, PersonRole.SUSPECT}


@dataclass
class _RoleMemory:
    """Persistent role state for one person."""

    role: PersonRole
    confidence: float
    event_id: int | None
    last_active_ms: float


class RoleResolver:
    """Assign roles with latching for confirmed events and grace period for tentative ones."""

    def __init__(self, config: RolesConfig | None = None):
        self._config = config
        self._memory: dict[int, _RoleMemory] = {}

    @property
    def config(self) -> RolesConfig:
        if self._config is None:
            self._config = RolesConfig()
        return self._config

    def reset(self) -> None:
        self._memory.clear()

    def update(
        self,
        observations: list[EntityObservation],
        events: list[SnatchEvent],
        person_label: str,
        timestamp_ms: float | None = None,
    ) -> dict[int, RoleAssignment]:
        person_ids = {
            item.entity_id
            for item in observations
            if item.label == person_label
        }

        # --- Step 1: compute fresh roles from current-frame events ---
        fresh: dict[int, RoleAssignment] = {}
        fresh_priorities: dict[int, int] = {}

        for event in events:
            if event.state == EventState.CANCELLED:
                continue
            priority = _STATE_PRIORITY[event.state]
            final = event.state == EventState.SUSPECTED
            victim_role = PersonRole.VICTIM if final else PersonRole.POSSIBLE_VICTIM
            suspect_role = PersonRole.SUSPECT if final else PersonRole.POSSIBLE_SUSPECT

            for pid, role in (
                (event.victim_person_id, victim_role),
                (event.suspect_person_id, suspect_role),
            ):
                if pid not in person_ids:
                    continue
                prev_priority = fresh_priorities.get(pid, 0)
                prev_confidence = fresh[pid].confidence if pid in fresh else -1.0
                if priority > prev_priority or (
                    priority == prev_priority and event.confidence > prev_confidence
                ):
                    fresh[pid] = RoleAssignment(
                        person_id=pid,
                        role=role,
                        confidence=event.confidence,
                        event_id=event.event_id,
                    )
                    fresh_priorities[pid] = priority

        # --- Step 2: merge with memory (latch + grace period) ---
        current_ms = timestamp_ms
        grace_ms = self.config.grace_period_seconds * 1000.0

        for pid in person_ids:
            fresh_role = fresh.get(pid)
            mem = self._memory.get(pid)

            if fresh_role is not None and fresh_role.role != PersonRole.NORMAL:
                # Active event is producing a role — update memory.
                self._memory[pid] = _RoleMemory(
                    role=fresh_role.role,
                    confidence=fresh_role.confidence,
                    event_id=fresh_role.event_id,
                    last_active_ms=current_ms or 0.0,
                )
            elif mem is not None:
                # No active event this frame — decide whether to keep memory.
                if mem.role in _CONFIRMED_ROLES and self.config.confirmed_lock:
                    # Confirmed roles are permanently latched.
                    pass
                elif current_ms is not None and mem.last_active_ms is not None:
                    if (current_ms - mem.last_active_ms) > grace_ms:
                        # Grace period expired — drop to NORMAL.
                        self._memory.pop(pid, None)
                # else: keep memory (within grace period)

        # --- Step 3: build final output ---
        roles: dict[int, RoleAssignment] = {}
        for pid in person_ids:
            mem = self._memory.get(pid)
            if mem is not None:
                roles[pid] = RoleAssignment(
                    person_id=pid,
                    role=mem.role,
                    confidence=mem.confidence,
                    event_id=mem.event_id,
                )
            else:
                identity_conf = next(
                    (item.identity_confidence for item in observations if item.entity_id == pid),
                    1.0,
                )
                roles[pid] = RoleAssignment(
                    person_id=pid,
                    role=PersonRole.NORMAL,
                    confidence=identity_conf,
                )

        # Tentative roles may be forgotten after their grace period, but a
        # confirmed role must survive a temporary camera occlusion.
        self._memory = {
            pid: mem
            for pid, mem in self._memory.items()
            if pid in person_ids
            or (mem.role in _CONFIRMED_ROLES and self.config.confirmed_lock)
            or current_ms is None
            or (current_ms - mem.last_active_ms) <= grace_ms
        }
        return roles
