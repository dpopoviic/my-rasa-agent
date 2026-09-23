"""
Custom actions for the Event Reservation App assistant.

None of these actions talk to a database directly. They are thin HTTP
clients for the internal API exposed by the .NET application
(Controllers/Api/InternalAgentController.cs) - the same application services
(IEventCatalogService, IMyReservationsService) that used to be called
in-process by the old Foundry agent tools are called here over HTTP instead.

Identity: the acting user id is read from the incoming message's metadata
(set by RasaChatbotConversationService on every call to the REST webhook),
never from a slot - a slot could in principle be influenced by what the user
types, metadata cannot. That id is forwarded as X-User-Id, alongside the
shared-secret X-Internal-Api-Key, so InternalApiKeyHandler on the .NET side
can build a ClaimsPrincipal for the request exactly as if it came from the
signed-in user's own browser.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

INTERNAL_API_BASE_URL = os.environ.get(
    "INTERNAL_API_BASE_URL", "https://localhost:5001"
).rstrip("/")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = 10
# false samo u razvoju (ASP.NET dev sertifikat); vidi .env
VERIFY_SSL = os.environ.get("INTERNAL_API_VERIFY_SSL", "true").lower() != "false"

NOT_SIGNED_IN_MESSAGE = "Morate biti prijavljeni da biste ovo uradili."
GENERIC_ERROR_MESSAGE = "Došlo je do problema prilikom komunikacije sa aplikacijom. Molimo pokušajte ponovo kasnije."


def _log_failure(exc: requests.RequestException) -> None:
    """Logs why a call to the .NET internal API failed (status, url, body)."""
    response = exc.response
    request = exc.request
    method = request.method if request is not None else "?"
    url = request.url if request is not None else "?"

    if response is not None:
        logger.warning(
            "Internal API call failed: %s %s -> HTTP %s, body: %s",
            method, url, response.status_code, (response.text or "")[:500],
        )
    else:
        logger.warning(
            "Internal API call failed (no response): %s %s -> %s: %s",
            method, url, type(exc).__name__, exc,
        )


def _current_user_id(tracker: Tracker) -> Optional[Text]:
    metadata = (tracker.latest_message or {}).get("metadata") or {}
    user_id = metadata.get("user_id")
    return user_id or None


def _headers(user_id: Text) -> Dict[Text, Text]:
    return {
        "X-Internal-Api-Key": INTERNAL_API_KEY,
        "X-User-Id": user_id,
    }


# Odgovori koji znace "bez filtera" - nisu pravi nazivi/lokacije.
_NO_FILTER_WORDS = {
    "ne", "nije bitno", "nebitno", "svejedno", "svi", "sve", "svi dogadjaji",
    "svi događaji", "bilo koji", "bilo gde", "bilo koje", "nista", "ništa",
    "nema", "-", "none", "null", "any", "all", "no",
}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _clean_text(value: Any) -> Optional[Text]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _NO_FILTER_WORDS:
        return None
    return text


def _clean_date(value: Any) -> Optional[Text]:
    """Samo ISO datum (YYYY-MM-DD...) ide dalje; sve drugo se ignorise."""
    text = _clean_text(value)
    return text if text and _ISO_DATE.match(text) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "da", "yes", "1"}


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ActionSearchEvents(Action):
    def name(self) -> Text:
        return "action_search_events"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_id = _current_user_id(tracker)
        if not user_id:
            return [SlotSet("events_search_result", NOT_SIGNED_IN_MESSAGE)]

        # Slotove popunjava LLM iz slobodnog teksta ("ne", "nije bitno", "svi
        # dogadjaji"), pa se ovde svode na prave filtere pre poziva .NET-a.
        params = {
            "searchTerm": _clean_text(tracker.get_slot("search_term")),
            "startDate": _clean_date(tracker.get_slot("start_date")),
            "endDate": _clean_date(tracker.get_slot("end_date")),
            "location": _clean_text(tracker.get_slot("location")),
            "availability": True if _as_bool(tracker.get_slot("only_available")) else None,
        }
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/events/search",
                params=params,
                headers=_headers(user_id),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            events = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("events_search_result", GENERIC_ERROR_MESSAGE)]

        if not events:
            return [SlotSet(
                "events_search_result",
                "Nema pronađenih događaja za zadate kriterijume.",
            )]

        lines = [
            "ID {eventId}: {name} - {location}, {start} do {end}, "
            "slobodno {available}/{capacity} mesta".format(
                eventId=e.get("eventId"),
                name=e.get("name"),
                location=e.get("location"),
                start=str(e.get("startDate", ""))[:10],
                end=str(e.get("endDate", ""))[:10],
                available=e.get("availablePlaces"),
                capacity=e.get("capacity"),
            )
            for e in events
        ]

        return [SlotSet("events_search_result", "\n".join(lines))]


class ActionGetEventAvailability(Action):
    def name(self) -> Text:
        return "action_get_event_availability"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_id = _current_user_id(tracker)
        if not user_id:
            return [SlotSet("event_availability_result", NOT_SIGNED_IN_MESSAGE)]

        event_id = _as_int(tracker.get_slot("event_id"))
        if event_id is None:
            return [SlotSet(
                "event_availability_result",
                "Nisam uspeo da prepoznam o kom događaju je reč. Možete li ponoviti naziv ili ID događaja?",
            )]

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/events/{event_id}/availability",
                headers=_headers(user_id),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            if response.status_code == 404:
                return [SlotSet(
                    "event_availability_result",
                    f"Nije pronađen događaj sa ID {event_id}.",
                )]
            response.raise_for_status()
            availability = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("event_availability_result", GENERIC_ERROR_MESSAGE)]

        message = (
            "{name}: {reserved}/{capacity} rezervisano, slobodno {available} mesta. "
            "{status}"
        ).format(
            name=availability.get("eventName"),
            reserved=availability.get("reservedCount"),
            capacity=availability.get("capacity"),
            available=availability.get("availablePlaces"),
            status=(
                "Rezervacije su moguće."
                if availability.get("isAvailableForReservation")
                else "Nema više slobodnih mesta."
            ),
        )

        return [SlotSet("event_availability_result", message)]


class ActionListMyReservations(Action):
    def name(self) -> Text:
        return "action_list_my_reservations"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_id = _current_user_id(tracker)
        if not user_id:
            return [SlotSet("my_reservations_result", NOT_SIGNED_IN_MESSAGE)]

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/reservations",
                headers=_headers(user_id),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            reservations = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("my_reservations_result", GENERIC_ERROR_MESSAGE)]

        if not reservations:
            return [SlotSet("my_reservations_result", "Trenutno nemate nijednu rezervaciju.")]

        lines = [
            "ID rezervacije {reservationId}: {eventName} ({eventStart}){notes}".format(
                reservationId=r.get("reservationId"),
                eventName=r.get("eventName"),
                eventStart=str(r.get("eventStartDate", ""))[:10],
                notes=f" - napomena: {r['notes']}" if r.get("notes") else "",
            )
            for r in reservations
        ]

        return [SlotSet("my_reservations_result", "\n".join(lines))]


class ActionReserveEvent(Action):
    def name(self) -> Text:
        return "action_reserve_event"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_id = _current_user_id(tracker)
        if not user_id:
            return [SlotSet("reservation_action_result", NOT_SIGNED_IN_MESSAGE)]

        event_id = _as_int(tracker.get_slot("event_id"))
        if event_id is None:
            return [SlotSet(
                "reservation_action_result",
                "Nisam uspeo da prepoznam koji događaj želite da rezervišete.",
            )]

        notes = tracker.get_slot("notes")

        try:
            response = requests.post(
                f"{INTERNAL_API_BASE_URL}/api/internal/reservations",
                json={"eventId": event_id, "notes": notes},
                headers=_headers(user_id),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        if result.get("success"):
            message = f"Rezervacija je uspešno kreirana (ID {result.get('reservationId')})."
        else:
            message = result.get("message") or "Rezervacija nije uspela."

        return [SlotSet("reservation_action_result", message)]


class ActionCancelReservation(Action):
    def name(self) -> Text:
        return "action_cancel_reservation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_id = _current_user_id(tracker)
        if not user_id:
            return [SlotSet("reservation_action_result", NOT_SIGNED_IN_MESSAGE)]

        reservation_id = _as_int(tracker.get_slot("reservation_id"))
        if reservation_id is None:
            return [SlotSet(
                "reservation_action_result",
                "Nisam uspeo da prepoznam koju rezervaciju želite da otkažete.",
            )]

        try:
            response = requests.delete(
                f"{INTERNAL_API_BASE_URL}/api/internal/reservations/{reservation_id}",
                headers=_headers(user_id),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        if result.get("success"):
            message = "Rezervacija je uspešno otkazana."
        else:
            message = result.get("message") or "Otkazivanje rezervacije nije uspelo."

        return [SlotSet("reservation_action_result", message)]
