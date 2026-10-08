"""MCP server sa alatima (samo citanje) za ReAct sub agenta event_assistant.

Alati su tanki HTTP klijenti za interni API .NET aplikacije
(Controllers/Api/InternalAgentController.cs), isto kao actions/actions.py, ali
ih poziva LLM agent preko MCP protokola umesto flow-a.

Identitet korisnika:
- Rasa agent salje token korisnika u skrivenom `_meta.user_token` polju svakog
  poziva alata (LLM ga ne vidi). Alat ga prosledjuje kao Authorization: Bearer,
  uz X-Internal-Api-Key, a .NET (InternalApiKeyHandler) ga ponovo proverava i
  korisnika uzima samo iz tokena.
- Sam MCP server prima samo zahteve sa ispravnim X-Mcp-Api-Key zaglavljem
  (Rasa ga salje iz endpoints.yml) i slusa samo na localhost-u.

Pokretanje (iz korena Rasa projekta):  python -m mcp_tools.server
"""

import calendar
import hmac
import json
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

load_dotenv()

logger = logging.getLogger("mcp_tools")

INTERNAL_API_BASE_URL = os.environ.get(
    "INTERNAL_API_BASE_URL", "https://localhost:5001"
).rstrip("/")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")
VERIFY_SSL = os.environ.get("INTERNAL_API_VERIFY_SSL", "true").lower() != "false"
REQUEST_TIMEOUT_SECONDS = 10

MCP_HOST = "127.0.0.1"  # samo lokalno - nikad ne izlagati spolja
MCP_PORT = int(os.environ.get("MCP_SERVER_PORT", "8765"))
MCP_API_KEY = os.environ.get("MCP_API_KEY", "")
MCP_API_KEY_HEADER = "x-mcp-api-key"

# Ogranicenje velicine odgovora: LLM dobija kompaktne podatke, ne ceo katalog.
MAX_EVENTS = 50
MAX_DESCRIPTION_LENGTH = 200

mcp = FastMCP("event-tools", host=MCP_HOST, port=MCP_PORT)


# ---------------------------------------------------------------------------
# Poziv .NET internog API-ja
# ---------------------------------------------------------------------------


class InternalApi:
    """HTTP klijent za /api/internal. `transport` postoji zbog testova."""

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._transport = transport

    async def get(
        self, path: str, user_token: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        async with httpx.AsyncClient(
            base_url=INTERNAL_API_BASE_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=VERIFY_SSL,
            transport=self._transport,
        ) as client:
            try:
                response = await client.get(
                    path,
                    params=params,
                    headers={
                        "X-Internal-Api-Key": INTERNAL_API_KEY,
                        "Authorization": f"Bearer {user_token}",
                    },
                )
            except httpx.HTTPError as exc:
                logger.warning("Internal API call failed: GET %s -> %r", path, exc)
                raise ToolError("The application is not reachable right now.") from exc

        if response.status_code == 404:
            return None
        if response.status_code in (401, 403):
            logger.warning("Internal API rejected the call: GET %s -> %s", path, response.status_code)
            raise ToolError("The user is not authorized for this data (session may have expired).")
        if response.is_error:
            logger.warning(
                "Internal API call failed: GET %s -> HTTP %s, body: %s",
                path, response.status_code, response.text[:500],
            )
            raise ToolError("The application returned an error.")
        return response.json()


api = InternalApi()


def _user_token(ctx: Context) -> str:
    """Token korisnika iz skrivenog _meta polja poziva (postavlja ga Rasa agent)."""
    meta = ctx.request_context.meta
    token = getattr(meta, "user_token", None) if meta is not None else None
    if not token:
        raise ToolError("The user is not signed in.")
    return token


def _to_json(value: Any) -> str:
    # Vraca se JSON tekst (ne Python dict) - LLM ga cita doslovno; cirilica ostaje citljiva.
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Filteri pretrage
# ---------------------------------------------------------------------------


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _parse_date(value: Optional[str], name: str) -> Optional[date]:
    text = _clean(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        raise ToolError(f"`{name}` must be a date in YYYY-MM-DD format, got '{value}'.")


def build_search_params(
    search_term: Optional[str] = None,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    only_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Pretvara bilo koju kombinaciju filtera u query parametre za /events/search.

    - nijedan filter: svi dogadjaji
    - year: cela godina; year + month: taj mesec
    - start_date / end_date: tacni datumi; imaju prednost nad granicom iz year/month
    - month bez year: greska (godina se ne pogadja)
    """
    if month is not None and not 1 <= month <= 12:
        raise ToolError("`month` must be between 1 and 12.")
    if month is not None and year is None:
        raise ToolError(
            "`month` requires `year`. Infer the year from the current date "
            "or ask the user which year they mean."
        )

    range_start: Optional[date] = None
    range_end: Optional[date] = None
    if year is not None:
        if month is not None:
            range_start = date(year, month, 1)
            range_end = date(year, month, calendar.monthrange(year, month)[1])
        else:
            range_start = date(year, 1, 1)
            range_end = date(year, 12, 31)

    range_start = _parse_date(start_date, "start_date") or range_start
    range_end = _parse_date(end_date, "end_date") or range_end
    if range_start and range_end and range_start > range_end:
        raise ToolError("`start_date` must not be after `end_date`.")

    params: Dict[str, Any] = {
        "searchTerm": _clean(search_term),
        "location": _clean(location),
        "startDate": range_start.isoformat() if range_start else None,
        "endDate": range_end.isoformat() if range_end else None,
        "availability": "true" if only_available else None,
    }
    return {k: v for k, v in params.items() if v is not None}


def _compact_event(e: Dict[str, Any]) -> Dict[str, Any]:
    description = e.get("description") or ""
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[:MAX_DESCRIPTION_LENGTH].rstrip() + "..."
    return {
        "event_id": e.get("eventId"),
        "name": e.get("name"),
        "location": e.get("location"),
        "start": e.get("startDate"),
        "end": e.get("endDate"),
        "capacity": e.get("capacity"),
        "available_places": e.get("availablePlaces"),
        "is_full": e.get("isFull"),
        "user_has_reservation": e.get("currentUserHasReservation"),
        "description": description,
    }


# ---------------------------------------------------------------------------
# Alati
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_events(
    ctx: Context,
    search_term: Optional[str] = None,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    only_available: Optional[bool] = None,
) -> str:
    """Search the event catalog. Every filter is optional and they can be combined;
    with no filters all events are returned.

    Args:
        search_term: Text matched against event name and description (Latin or Cyrillic).
        location: City or venue, in nominative case (e.g. "Beograd", not "u Beogradu").
        start_date: Only events starting on or after this date (YYYY-MM-DD).
        end_date: Only events starting on or before this date (YYYY-MM-DD).
        year: Restrict to this year (e.g. 2026). Explicit start_date/end_date override its bounds.
        month: Month 1-12; requires `year`.
        only_available: True to return only events that still have free places.

    Returns JSON {count, truncated, events[]} sorted by start time. Each event has
    event_id, name, location, start and end (ISO date-time), capacity,
    available_places, is_full, user_has_reservation and a shortened description.
    """
    params = build_search_params(
        search_term, location, start_date, end_date, year, month, only_available
    )
    events: List[Dict[str, Any]] = await api.get(
        "/api/internal/events/search", _user_token(ctx), params
    ) or []
    return _to_json({
        "count": len(events),
        "truncated": len(events) > MAX_EVENTS,
        "events": [_compact_event(e) for e in events[:MAX_EVENTS]],
    })


@mcp.tool()
async def get_event_availability(ctx: Context, event_id: int) -> str:
    """Current capacity of one event: reserved count, free places and whether it
    can still be reserved. Use the event_id returned by search_events.
    """
    availability = await api.get(
        f"/api/internal/events/{event_id}/availability", _user_token(ctx)
    )
    if availability is None:
        raise ToolError(f"No event with event_id {event_id}.")
    return _to_json({
        "event_id": availability.get("eventId"),
        "name": availability.get("eventName"),
        "capacity": availability.get("capacity"),
        "reserved_count": availability.get("reservedCount"),
        "available_places": availability.get("availablePlaces"),
        "can_reserve": availability.get("isAvailableForReservation"),
    })


@mcp.tool()
async def get_my_reservations(ctx: Context) -> str:
    """All reservations of the signed-in user, with each event's start, end and
    location. Returns JSON {count, reservations[]}.
    """
    reservations: List[Dict[str, Any]] = await api.get(
        "/api/internal/reservations", _user_token(ctx)
    ) or []
    return _to_json({
        "count": len(reservations),
        "reservations": [
            {
                "reservation_id": r.get("reservationId"),
                "event_id": r.get("eventId"),
                "event_name": r.get("eventName"),
                "location": r.get("eventLocation"),
                "start": r.get("eventStartDate"),
                "end": r.get("eventEndDate"),
                "notes": r.get("notes"),
            }
            for r in reservations
        ],
    })


# ---------------------------------------------------------------------------
# HTTP aplikacija sa proverom MCP API kljuca
# ---------------------------------------------------------------------------


def require_api_key(app: Any, api_key: str) -> Any:
    """ASGI omotac: HTTP zahtev bez ispravnog X-Mcp-Api-Key dobija 401."""
    expected = api_key.encode()

    async def wrapped(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            provided = headers.get(MCP_API_KEY_HEADER.encode(), b"")
            if not hmac.compare_digest(provided, expected):
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({"type": "http.response.body", "body": b'{"error":"invalid api key"}'})
                return
        await app(scope, receive, send)  # lifespan i ispravni zahtevi idu dalje

    return wrapped


def create_app() -> Any:
    if not MCP_API_KEY:
        raise RuntimeError("MCP_API_KEY is not set (.env).")
    return require_api_key(mcp.streamable_http_app(), MCP_API_KEY)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=MCP_HOST, port=MCP_PORT)
