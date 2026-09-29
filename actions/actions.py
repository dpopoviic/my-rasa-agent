"""
Custom actions for the Event Reservation App assistant.

None of these actions talk to a database directly. They are thin HTTP
clients for the internal API exposed by the .NET application
(Controllers/Api/InternalAgentController.cs) - the same application services
(IEventCatalogService, IMyReservationsService) that used to be called
in-process by the old Foundry agent tools are called here over HTTP instead.

Identity: the acting user is proven by a signed user token that the .NET app
issues for every chat message (UserTokenService). The secure_rest channel
(secure_rest_channel.py) verifies it and puts it into the message metadata;
it is read from there, never from a slot - a slot could in principle be
influenced by what the user types, metadata cannot. The token is forwarded as
Authorization: Bearer, alongside the shared-secret X-Internal-Api-Key, and
InternalApiKeyHandler on the .NET side verifies it again and takes the user id
only from the token, so these actions cannot act for any other user.
"""
import logging
import calendar
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

NOT_SIGNED_IN_MESSAGE = "Морате бити пријављени да бисте ово урадили."
GENERIC_ERROR_MESSAGE = "Дошло је до проблема приликом комуникације са апликацијом. Молимо покушајте поново касније."


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


def _current_user_token(tracker: Tracker) -> Optional[Text]:
    # Metadata upisuje secure_rest kanal tek posto proveri token; poslednja poruka
    # uvek nosi svez token (novi za svaku poruku), pa se ne cuva u slotu.
    metadata = (tracker.latest_message or {}).get("metadata") or {}
    user_token = metadata.get("user_token")
    return user_token or None


def _headers(user_token: Text) -> Dict[Text, Text]:
    return {
        "X-Internal-Api-Key": INTERNAL_API_KEY,
        "Authorization": f"Bearer {user_token}",
    }


# Odgovori koji znace "bez filtera" - nisu pravi nazivi/lokacije.
_NO_FILTER_WORDS = {
    "ne", "nije bitno", "nebitno", "svejedno", "svi", "sve", "svi dogadjaji",
    "svi događaji", "bilo koji", "bilo gde", "bilo koje", "nista", "ništa",
    "nema", "-", "none", "null", "any", "all", "no", "nemam", "bez napomene", "ne hvala",
    # isto na cirilici - korisnik moze da pise i cirilicom
    "не", "није битно", "небитно", "свеједно", "сви", "све", "сви догађаји",
    "било који", "било где", "било које", "ништа", "нема", "немам", "без напомене", "не хвала",
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
    return str(value).strip().lower() in {"true", "da", "да", "yes", "1"}


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _norm(value: Any) -> Text:
    return str(value or "").strip().casefold()

def _event_found(event: Dict[Text, Any], status: Text) -> List[Dict[Text, Any]]:
    return[
        SlotSet("event_id", event.get("eventId")),
        SlotSet("event_name", event.get("name")), #pun naziv za potvrdu i poruke
        SlotSet("event_match", status),
    ]

class ActionSearchEvents(Action):
    def name(self) -> Text:
        return "action_search_events"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_token = _current_user_token(tracker)
        if not user_token:
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

        # Mesec/godina -> opseg datuma; konkretni datumi koje je korisnik naveo imaju prednost.
        month = _as_int(_clean_text(tracker.get_slot("search_month")))
        year = _as_int(_clean_text(tracker.get_slot("search_year")))
        if year and "startDate" not in params and "endDate" not in params:
            if month and 1 <= month <= 12:
                last_day = calendar.monthrange(year, month)[1]
                params["startDate"] = f"{year}-{month:02d}-01"
                params["endDate"] = f"{year}-{month:02d}-{last_day:02d}"
            else:
                params["startDate"] = f"{year}-01-01"
                params["endDate"] = f"{year}-12-31"

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/events/search",
                params=params,
                headers=_headers(user_token),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            events = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("events_search_result", GENERIC_ERROR_MESSAGE)]

        if not events:
            return [
                SlotSet("events_search_result", "Нема пронађених догађаја за задате критеријуме."),
                SlotSet("last_listed_events", []),
            ]
        
        lines = [
            "{name} - {location}, {start} до {end}, "
            "слободно {available}/{capacity} места".format(
                name=e.get("name"),
                location=e.get("location"),
                start=str(e.get("startDate", ""))[:10],
                end=str(e.get("endDate", ""))[:10],
                available=e.get("availablePlaces"),
                capacity=e.get("capacity"),
            )

            for e in events
        ]

        return [
            SlotSet("events_search_result", "\n".join(lines)),
            SlotSet("last_listed_events", [e.get("name") for e in events]),
        ]

class ActionResolveEvent(Action):
    """Pretvara naziv (ili deo naziva) koji je korisnik rekao u tacno jedan dogadjaj.

    Poklapanje naziva (cirilica/latinica, samo buduci dogadjaji) radi .NET
    (/events/resolve); ovde se odlucuje samo o toku razgovora.
    """

    def name(self) -> Text:
        return "action_resolve_event"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_token = _current_user_token(tracker)
        if not user_token:
            dispatcher.utter_message(text=NOT_SIGNED_IN_MESSAGE)
            return [SlotSet("event_match", "error")]

        term = _clean_text(tracker.get_slot("event_name"))
        if not term:
            return [SlotSet("event_match", "not_found"), SlotSet("event_name", None)]

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/events/resolve",
                params={"name": term},
                headers=_headers(user_token),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            candidates = response.json() or []
        except requests.RequestException as exc:
            _log_failure(exc)
            dispatcher.utter_message(text=GENERIC_ERROR_MESSAGE)
            return [SlotSet("event_match", "error")]

        if len(candidates) == 1:
            return _event_found(candidates[0], "found")

        if not candidates:
            dispatcher.utter_message(
                text=f"Нисам пронашао догађај под називом „{term}“. Молим унесите тачан назив."
            )
            return [SlotSet("event_match", "not_found"), SlotSet("event_name", None)]

        # Vise kandidata: ako je tacno jedan bio u poslednjem prikazanom spisku, trazimo potvrdu.
        shown = {_norm(n) for n in (tracker.get_slot("last_listed_events") or [])}
        from_last = [c for c in candidates if _norm(c.get("name")) in shown]
        if len(from_last) == 1:
            return _event_found(from_last[0], "confirm")

        names = "\n".join(
            f"- {c.get('name')} ({str(c.get('startDate', ''))[:10]})" for c in candidates
        )
        dispatcher.utter_message(
            text=f"Пронашао сам више догађаја који садрже „{term}“:\n{names}"
        )
        return [SlotSet("event_match", "ambiguous"), SlotSet("event_name", None)]

class ActionGetEventAvailability(Action):
    def name(self) -> Text:
        return "action_get_event_availability"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_token = _current_user_token(tracker)
        if not user_token:
            return [SlotSet("event_availability_result", NOT_SIGNED_IN_MESSAGE)]

        event_id = _as_int(tracker.get_slot("event_id"))
        if event_id is None:
            return [SlotSet(
                "event_availability_result",
                "Нисам успео да препознам о ком догађају је реч. Можете ли поновити пун назив догађаја?",
            )]

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/events/{event_id}/availability",
                headers=_headers(user_token),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            if response.status_code == 404:
                return [SlotSet(
                    "event_availability_result",
                    "Тражени догађај није пронађен.",
                )]
            response.raise_for_status()
            availability = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("event_availability_result", GENERIC_ERROR_MESSAGE)]

        message = (
            "{name}: {reserved}/{capacity} резервисано, слободно {available} места. "
            "{status}"
        ).format(
            name=availability.get("eventName"),
            reserved=availability.get("reservedCount"),
            capacity=availability.get("capacity"),
            available=availability.get("availablePlaces"),
            status=(
                "Резервације су могуће."
                if availability.get("isAvailableForReservation")
                else "Нема више слободних места."
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
        user_token = _current_user_token(tracker)
        if not user_token:
            return [SlotSet("my_reservations_result", NOT_SIGNED_IN_MESSAGE)]

        try:
            response = requests.get(
                f"{INTERNAL_API_BASE_URL}/api/internal/reservations",
                headers=_headers(user_token),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            reservations = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("my_reservations_result", GENERIC_ERROR_MESSAGE)]

        if not reservations:
            return [SlotSet("my_reservations_result", "Тренутно немате ниједну резервацију.")]

        lines = [
            "ID резервације {reservationId}: {eventName} ({eventStart}){notes}".format(
                reservationId=r.get("reservationId"),
                eventName=r.get("eventName"),
                eventStart=str(r.get("eventStartDate", ""))[:10],
                notes=f" - напомена: {r['notes']}" if r.get("notes") else "",
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
        user_token = _current_user_token(tracker)
        if not user_token:
            return [SlotSet("reservation_action_result", NOT_SIGNED_IN_MESSAGE)]

        event_id = _as_int(tracker.get_slot("event_id"))
        if event_id is None:
            return [SlotSet(
                "reservation_action_result",
                "Нисам успео да препознам који догађај желите да резервишете.",
            )]

        notes = _clean_text(tracker.get_slot("notes"))

        try:
            response = requests.post(
                f"{INTERNAL_API_BASE_URL}/api/internal/reservations",
                json={"eventId": event_id, "notes": notes},
                headers=_headers(user_token),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        if result.get("success"):
            message = f"Резервација за догађај {tracker.get_slot('event_name')} је успешно креирана."
        else:
            message = result.get("message") or "Резервација није успела."

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
        user_token = _current_user_token(tracker)
        if not user_token:
            return [SlotSet("reservation_action_result", NOT_SIGNED_IN_MESSAGE)]

        reservation_id = _as_int(tracker.get_slot("reservation_id"))
        if reservation_id is None:
            return [SlotSet(
                "reservation_action_result",
                "Нисам успео да препознам коју резервацију желите да откажете.",
            )]

        try:
            response = requests.delete(
                f"{INTERNAL_API_BASE_URL}/api/internal/reservations/{reservation_id}",
                headers=_headers(user_token),
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=VERIFY_SSL,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            _log_failure(exc)
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        if result.get("success"):
            message = "Резервација је успешно отказана."
        else:
            message = result.get("message") or "Отказивање резервације није успело."

        return [SlotSet("reservation_action_result", message)]
