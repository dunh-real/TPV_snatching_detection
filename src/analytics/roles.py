"""Resolve event-scoped hypotheses into one display role per person."""

from src.analytics.models import (
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


class RoleResolver:
    """Assign roles without treating a tentative hypothesis as a conviction."""

    def reset(self) -> None:
        # Kept as an explicit lifecycle hook for a stable facade API.
        return None

    def update(
        self,
        observations: list[EntityObservation],
        events: list[SnatchEvent],
        person_label: str,
    ) -> dict[int, RoleAssignment]:
        roles = {
            item.entity_id: RoleAssignment(
                person_id=item.entity_id,
                role=PersonRole.NORMAL,
                confidence=item.identity_confidence,
            )
            for item in observations
            if item.label == person_label
        }
        priorities = {person_id: 0 for person_id in roles}

        for event in events:
            if event.state == EventState.CANCELLED:
                continue
            priority = _STATE_PRIORITY[event.state]
            final = event.state == EventState.SUSPECTED
            victim_role = PersonRole.VICTIM if final else PersonRole.POSSIBLE_VICTIM
            suspect_role = PersonRole.SUSPECT if final else PersonRole.POSSIBLE_SUSPECT
            self._assign(
                roles,
                priorities,
                event.victim_person_id,
                victim_role,
                event,
                priority,
            )
            self._assign(
                roles,
                priorities,
                event.suspect_person_id,
                suspect_role,
                event,
                priority,
            )
        return roles

    @staticmethod
    def _assign(
        roles: dict[int, RoleAssignment],
        priorities: dict[int, int],
        person_id: int,
        role: PersonRole,
        event: SnatchEvent,
        priority: int,
    ) -> None:
        if person_id not in roles or priority < priorities[person_id]:
            return
        current = roles[person_id]
        if priority == priorities[person_id] and event.confidence <= current.confidence:
            return
        roles[person_id] = RoleAssignment(
            person_id=person_id,
            role=role,
            confidence=event.confidence,
            event_id=event.event_id,
        )
        priorities[person_id] = priority
