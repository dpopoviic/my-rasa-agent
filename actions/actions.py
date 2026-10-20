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

Concurrency: the actions run inside the Rasa process (actions_module), on its
single asyncio event loop, so they are async and await every call to .NET. A
blocking call would stop Rasa for every user until .NET answers. All calls share
one HTTP client, so connections (and TLS sessions) to .NET are reused.

Logging: a failed call logs only the method, path (no query string) and status -
never the response body or the search parameters, which can hold users' data.

Retries: a write (reserve, cancel) whose answer is lost - connection dropped,
timeout, .NET 5xx - is sent again, always with the same Idempotency-Key, so .NET
(IdempotencyService) answers the repeat from what it stored instead of running it
twice. The key is built from the message id (claim jti), the action and the
resource, so every new user message gets new keys and runs for real.
"""
import asyncio
import logging
import calendar
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Text

import httpx
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

INTERNAL_API_BASE_URL = os.environ.get(
    "INTERNAL_API_BASE_URL", "https://localhost:5001"
).rstrip("/")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = 10

VERIFY_SSL = os.environ.get("INTERNAL_API_VERIFY_SSL", "true").lower() != "false"

NOT_SIGNED_IN_MESSAGE = "Морате бити пријављени да бисте ово урадили."
GENERIC_ERROR_MESSAGE = "Дошло је до проблема приликом комуникације са апликацијом. Молимо покушајте поново касније."

# Kodovi koje .NET vraca u ReservationOperationResult.errorCode -> poruka za korisnika.
# Akcija rezervacije AlreadyReserved obradjuje sama (rezervacija postoji, pa to nije greska).
_DOTNET_ERROR_CODES = {
    "EventNotFound": "Изабрани догађај није пронађен.",
    "AlreadyReserved": "Већ имате резервацију за овај догађај.",
    "EventFull": "Догађај је попуњен и више не прима резервације.",
    "ReservationNotFound": "Резервација са тим ID-јем не постоји међу Вашим резервацијама.",
}

# Rezerva za .NET bez errorCode (starija verzija): poruka (engleski) -> za korisnika.
# Moze da se obrise kad je .NET sa errorCode pusten svuda.
_DOTNET_ERROR_MESSAGES = {
    "Reservation not found.": "Резервација са тим ID-јем не постоји међу Вашим резервацијама.",
    "The selected event could not be found.": "Изабрани догађај није пронађен.",
    "You already have a reservation for this event.": "Већ имате резервацију за овај догађај.",
    "This event has reached its capacity and can no longer accept reservations.":
        "Догађај је попуњен и више не прима резервације.",
}

# Greske poziva .NET-a: mreza/timeout/HTTP status (httpx.HTTPError) ili telo koje nije JSON
CALL_ERRORS = (httpx.HTTPError, ValueError)

# Upisni poziv (rezervacija, otkazivanje) se pokusava najvise toliko puta, sa pauzama
# izmedju pokusaja. Ukupno (3 x 10 s + 2 s) daleko je ispod roka tokena (5 min).
WRITE_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (0.5, 1.5)

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
# Fiksan namespace za uuid5: isti ulaz (poruka, akcija, resurs) uvek daje isti kljuc.
# Ne menjati - promena bi dala druge kljuceve za iste zahteve.
IDEMPOTENCY_NAMESPACE = uuid.UUID("6f1c2a3e-8d4b-4e6a-9c1f-2b7d5e8a9c30")

_client: Optional[httpx.AsyncClient] = None

def _http() -> httpx.AsyncClient:
    """Jedan HTTP klijent za ceo Rasa proces (connection pooling).

    Pravi se pri prvom pozivu, unutar Rasinog event loop-a, za koji se vezuje.
    Testovi ga zamenjuju klijentom sa laznim transportom.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=INTERNAL_API_BASE_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=VERIFY_SSL,
        )
    return _client

async def _call_json(method: Text, path: Text, user_token: Text, **kwargs: Any) -> Any:
    """Poziv .NET internog API-ja; vraca JSON telo ili baca neku od CALL_ERRORS."""
    response = await _http().request(method, path, headers=_headers(user_token), **kwargs)
    response.raise_for_status()
    return response.json()

def _is_retryable(exc: Exception) -> bool:
    """Da li ponovljen zahtev ima smisla: odgovor nije stigao, ili je .NET pao (5xx).

    4xx se ne ponavlja - isti zahtev bi dobio isti odgovor. Telo koje nije JSON
    (ValueError) takodje ne: .NET je odgovorio, samo necim neocekivanim.
    """
    if isinstance(exc, httpx.TransportError):  # veza odbijena/pukla, timeout
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False

async def _call_json_idempotent(
    method: Text, path: Text, user_token: Text, idempotency_key: Text, tracker: Tracker,
    **kwargs: Any,
) -> Any:
    """Upisni poziv .NET-a sa Idempotency-Key i ponavljanjem.

    Svaki pokusaj salje ISTI kljuc: ako je .NET prvi zahtev vec izvrsio (a odgovor
    se izgubio), ponovljeni dobija zapamcen odgovor i nista se ne izvrsava dvaput.
    Posle poslednjeg neuspelog pokusaja baca gresku kao _call_json.
    """
    headers = {**_headers(user_token), IDEMPOTENCY_KEY_HEADER: idempotency_key}
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            response = await _http().request(method, path, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except CALL_ERRORS as exc:
            if attempt == WRITE_ATTEMPTS or not _is_retryable(exc):
                raise
            reason = (
                f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            logger.warning(
                "Internal API call failed, retrying (%s/%s): %s %s -> %s (correlation_id=%s)",
                attempt, WRITE_ATTEMPTS, method, path, reason, _correlation_id(tracker),
            )
            await asyncio.sleep(RETRY_DELAYS_SECONDS[attempt - 1])

async def _has_reservation_for_event(user_token: Text, event_id: int, tracker: Tracker) -> bool:
    """Provera stanja posle neuspelog upisa: da li korisnik ipak ima rezervaciju za dogadjaj.

    False i kad ni ova provera ne uspe - tada se ne zna, pa korisnik dobija poruku o gresci.
    """
    try:
        reservations = await _call_json("GET", "/api/internal/reservations", user_token) or []
    except CALL_ERRORS as exc:
        _log_failure(exc, tracker)
        return False
    return any(_as_int(r.get("eventId")) == event_id for r in reservations)

def _idempotency_key(tracker: Tracker, action_name: Text, resource_id: int) -> Optional[Text]:
    """Kljuc za jedan upis: ista poruka + ista akcija + isti resurs = isti kljuc.

    message_id (claim jti) je nov za svaku poruku korisnika, pa nova poruka (nova
    namera) dobija nov kljuc i izvrsava se, a ponovljen poziv iz iste poruke ne.
    None ako poruka nema message_id - tada se kljuc ne moze napraviti.
    """
    message_id = _message_metadata(tracker).get("message_id")
    if not message_id:
        return None
    return str(uuid.uuid5(IDEMPOTENCY_NAMESPACE, f"{message_id}:{action_name}:{resource_id}"))

def _log_failure(exc: Exception, tracker: Tracker) -> None:
    """Zasto poziv .NET-a nije uspeo - bez tela odgovora i bez query stringa.

    correlation_id povezuje ovaj log sa logovima .NET-a i Rase za istu poruku.
    """
    correlation_id = _correlation_id(tracker)
    if isinstance(exc, httpx.HTTPStatusError):
        logger.warning(
            "Internal API call failed: %s %s -> HTTP %s (correlation_id=%s)",
            exc.request.method, exc.request.url.path, exc.response.status_code, correlation_id,
        )
    elif isinstance(exc, httpx.RequestError):
        logger.warning(
            "Internal API call failed (no response): %s %s -> %s (correlation_id=%s)",
            exc.request.method, exc.request.url.path, type(exc).__name__, correlation_id,
        )
    else:
        logger.warning(
            "Internal API call failed: %s (correlation_id=%s)", type(exc).__name__, correlation_id
        )

def _operation_error(result: Dict[Text, Any], default: Text) -> Text:
    """Poruka za korisnika kad .NET vrati success: false - po errorCode, a po tekstu samo ako koda nema."""
    code = result.get("errorCode")
    if code:
        return _DOTNET_ERROR_CODES.get(code, default)
    return _DOTNET_ERROR_MESSAGES.get(result.get("errorMessage") or "", default)

def _message_metadata(tracker: Tracker) -> Dict[Text, Any]:
    # Metadata upisuje secure_rest kanal tek posto proveri token; poslednja poruka
    # uvek nosi svez token (novi za svaku poruku), pa se nista od ovoga ne cuva u slotu.
    return (tracker.latest_message or {}).get("metadata") or {}

def _current_user_token(tracker: Tracker) -> Optional[Text]:
    return _message_metadata(tracker).get("user_token") or None

def _correlation_id(tracker: Tracker) -> Text:
    """correlation_id poruke (claim cid iz tokena); "-" ako ga nema, da se log ipak ispise."""
    return _message_metadata(tracker).get("correlation_id") or "-"

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

    async def run(
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
            events = await _call_json(
                "GET", "/api/internal/events/search", user_token, params=params
            )
        except CALL_ERRORS as exc:
            _log_failure(exc, tracker)
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

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_token = _current_user_token(tracker)
        if not user_token:
            dispatcher.utter_message(response="utter_not_signed_in")
            return [SlotSet("event_match", "error")]

        term = _clean_text(tracker.get_slot("event_name"))
        if not term:
            return [SlotSet("event_match", "not_found"), SlotSet("event_name", None)]

        try:
            candidates = await _call_json(
                "GET", "/api/internal/events/resolve", user_token, params={"name": term}
            ) or []
        except CALL_ERRORS as exc:
            _log_failure(exc, tracker)
            dispatcher.utter_message(response="utter_generic_error")
            return [SlotSet("event_match", "error")]

        if len(candidates) == 1:
            return _event_found(candidates[0], "found")

        if not candidates:
            dispatcher.utter_message(response="utter_event_name_not_found", term=term)
            return [SlotSet("event_match", "not_found"), SlotSet("event_name", None)]

        # Vise kandidata: ako je tacno jedan bio u poslednjem prikazanom spisku, trazimo potvrdu.
        shown = {_norm(n) for n in (tracker.get_slot("last_listed_events") or [])}
        from_last = [c for c in candidates if _norm(c.get("name")) in shown]
        if len(from_last) == 1:
            return _event_found(from_last[0], "confirm")

        names = "\n".join(
            f"- {c.get('name')} ({str(c.get('startDate', ''))[:10]})" for c in candidates
        )
        dispatcher.utter_message(response="utter_multiple_events_found", term=term, names=names)
        return [SlotSet("event_match", "ambiguous"), SlotSet("event_name", None)]

class ActionGetEventAvailability(Action):
    def name(self) -> Text:
        return "action_get_event_availability"

    async def run(
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
            availability = await _call_json(
                "GET", f"/api/internal/events/{event_id}/availability", user_token
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return [SlotSet(
                    "event_availability_result",
                    "Тражени догађај није пронађен.",
                )]
            _log_failure(exc, tracker)
            return [SlotSet("event_availability_result", GENERIC_ERROR_MESSAGE)]
        except CALL_ERRORS as exc:
            _log_failure(exc, tracker)
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

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_token = _current_user_token(tracker)
        if not user_token:
            return [SlotSet("my_reservations_result", NOT_SIGNED_IN_MESSAGE)]

        try:
            reservations = await _call_json("GET", "/api/internal/reservations", user_token)
        except CALL_ERRORS as exc:
            _log_failure(exc, tracker)
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

    async def run(
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

        idempotency_key = _idempotency_key(tracker, self.name(), event_id)
        if not idempotency_key:
            # Bez kljuca .NET odbija upis (400); kanal uvek upisuje message_id, pa je ovo greska
            logger.warning(
                "Reservation not sent: message has no message_id (correlation_id=%s)",
                _correlation_id(tracker),
            )
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        try:
            result = await _call_json_idempotent(
                "POST", "/api/internal/reservations", user_token, idempotency_key, tracker,
                json={"eventId": event_id, "notes": notes},
            )
        except CALL_ERRORS as exc:
            _log_failure(exc, tracker)
            # Svi pokusaji su pali, ali je .NET mozda ipak upisao (odgovor se izgubio):
            # pre poruke o gresci proverava se da li rezervacija postoji.
            if await _has_reservation_for_event(user_token, event_id, tracker):
                return [SlotSet(
                    "reservation_action_result",
                    f"Резервација за догађај {tracker.get_slot('event_name')} је активна.",
                )]
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        if result.get("success"):
            message = f"Резервација за догађај {tracker.get_slot('event_name')} је успешно креирана."
        elif result.get("errorCode") == "AlreadyReserved":
            # Korisnik vec ima rezervaciju (ranije, ili ju je upisao paralelni/ponovljeni
            # zahtev) - ishod koji je trazio postoji, pa se ne prikazuje kao greska.
            message = f"Већ имате резервацију за догађај {tracker.get_slot('event_name')} - резервација је активна."
        else:
            message = _operation_error(result, "Резервација није успела.")

        return [SlotSet("reservation_action_result", message)]

class ActionCancelReservation(Action):
    def name(self) -> Text:
        return "action_cancel_reservation"

    async def run(
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

        idempotency_key = _idempotency_key(tracker, self.name(), reservation_id)
        if not idempotency_key:
            logger.warning(
                "Cancellation not sent: message has no message_id (correlation_id=%s)",
                _correlation_id(tracker),
            )
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        try:
            # Sa kljucem ponovljeno otkazivanje (posle izgubljenog odgovora) dobija
            # prvobitni uspeh, a ne "Reservation not found".
            result = await _call_json_idempotent(
                "DELETE", f"/api/internal/reservations/{reservation_id}", user_token,
                idempotency_key, tracker,
            )
        except CALL_ERRORS as exc:
            _log_failure(exc, tracker)
            return [SlotSet("reservation_action_result", GENERIC_ERROR_MESSAGE)]

        if result.get("success"):
            message = "Резервација је успешно отказана."
        else:
            message = _operation_error(result, "Отказивање резервације није успело.")

        return [SlotSet("reservation_action_result", message)]

# Vrednosti slota requested_language (domain.yml) -> kodovi jezika iz config.yml
REQUESTED_LANGUAGE_CODES = {
    "serbian_cyrillic": "sr-Cyrl",
    "serbian_latin": "sr-Latn",
    "english": "en",
}

class ActionSetLanguage(Action):
    """Flow change_language: postavlja ugradjeni slot `language` na jezik koji je
    korisnik izricito trazio. Rephraser, EnterpriseSearch i event_assistant ga citaju
    iz tog slota. Na pocetku nove sesije Rasa ga ponovo uzima iz metadata aplikacije."""

    def name(self) -> Text:
        return "action_set_language"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        language = REQUESTED_LANGUAGE_CODES.get(tracker.get_slot("requested_language"))
        # requested_language se brise da bi sledeci zahtev za promenu ponovo bio popunjen
        events = [SlotSet("requested_language", None)]
        if language:
            events.insert(0, SlotSet("language", language))
        return events
