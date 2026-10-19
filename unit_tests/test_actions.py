"""Testovi custom akcija (actions/actions.py) - .NET interni API je zamenjen laznim
httpx transportom, nista ne mora da radi.

Pokrivaju: da akcije ne blokiraju Rasin event loop, da salju prave kredencijale,
da korisnik dobija razumljivu poruku za greske iz .NET-a i da logovi ne sadrze
podatke korisnika (telo odgovora, parametre pretrage).

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import asyncio
import inspect
import json
import time
import unittest
from typing import Any, Dict, List, Optional

import httpx
from rasa_sdk.executor import CollectingDispatcher

from actions import actions

TOKEN = "user-token-123"
CORRELATION_ID = "correlation-1"


class FakeTracker:
    """Samo ono sto akcije citaju: metadata poslednje poruke i slotovi."""

    def __init__(self, token: Optional[str] = TOKEN, slots: Optional[Dict[str, Any]] = None,
                 correlation_id: Optional[str] = CORRELATION_ID):
        metadata: Dict[str, Any] = {}
        if token:
            metadata["user_token"] = token
        if correlation_id:
            metadata["correlation_id"] = correlation_id
        self.latest_message = {"metadata": metadata}
        self._slots = slots or {}

    def get_slot(self, key: str) -> Any:
        return self._slots.get(key)


class FakeDotNet:
    """Lazni .NET: pamti zahteve i vraca zadat status i telo (opciono sa kasnjenjem)."""

    def __init__(self, status: int = 200, body: Any = None, delay: float = 0.0,
                 raw_body: Optional[bytes] = None, error: Optional[Exception] = None):
        self.status, self.body, self.delay = status, body, delay
        self.raw_body, self.error = raw_body, error
        self.requests: List[httpx.Request] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        if self.raw_body is not None:
            return httpx.Response(self.status, content=self.raw_body)
        return httpx.Response(self.status, json=self.body)


def slot_value(events: List[Dict[str, Any]], name: str) -> Any:
    return next(e["value"] for e in events if e.get("name") == name)


class ActionTestCase(unittest.IsolatedAsyncioTestCase):
    def use_fake(self, fake: FakeDotNet) -> FakeDotNet:
        original = actions._client
        actions._client = httpx.AsyncClient(
            base_url="https://dotnet.test", transport=httpx.MockTransport(fake.handler)
        )
        self.addAsyncCleanup(actions._client.aclose)
        self.addCleanup(setattr, actions, "_client", original)
        return fake

    async def run_action(self, action_cls, tracker: FakeTracker) -> List[Dict[str, Any]]:
        return await action_cls().run(CollectingDispatcher(), tracker, {})


class ConcurrencyTests(ActionTestCase):
    def test_actions_calling_dotnet_are_async(self):
        # Sinhrona akcija bi blokirala Rasin event loop za sve korisnike
        for action_cls in (
            actions.ActionSearchEvents, actions.ActionResolveEvent,
            actions.ActionGetEventAvailability, actions.ActionListMyReservations,
            actions.ActionReserveEvent, actions.ActionCancelReservation,
        ):
            self.assertTrue(inspect.iscoroutinefunction(action_cls.run), action_cls.__name__)

    async def test_waiting_for_dotnet_does_not_block_other_work(self):
        self.use_fake(FakeDotNet(body=[], delay=0.5))
        started = time.monotonic()

        async def other_user() -> float:
            await asyncio.sleep(0.01)
            return time.monotonic() - started

        _, other_finished_after = await asyncio.gather(
            self.run_action(actions.ActionListMyReservations, FakeTracker()), other_user()
        )

        # Drugi posao se zavrsio dok je akcija jos cekala .NET (0,5 s)
        self.assertLess(other_finished_after, 0.3)

    def test_http_client_is_shared(self):
        original = actions._client
        actions._client = None
        try:
            first = actions._http()
            self.assertIs(actions._http(), first)
            self.assertEqual(str(first.base_url).rstrip("/"), actions.INTERNAL_API_BASE_URL)
        finally:
            actions._client = original


class CredentialTests(ActionTestCase):
    async def test_user_token_and_api_key_are_sent(self):
        fake = self.use_fake(FakeDotNet(body=[]))

        await self.run_action(actions.ActionListMyReservations, FakeTracker())

        request = fake.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.path, "/api/internal/reservations")
        self.assertEqual(request.headers["Authorization"], f"Bearer {TOKEN}")
        self.assertIn("X-Internal-Api-Key", request.headers)

    async def test_without_token_dotnet_is_not_called(self):
        fake = self.use_fake(FakeDotNet(body=[]))

        events = await self.run_action(actions.ActionListMyReservations, FakeTracker(token=None))

        self.assertEqual(fake.requests, [])
        self.assertEqual(slot_value(events, "my_reservations_result"), actions.NOT_SIGNED_IN_MESSAGE)


class ActionResultTests(ActionTestCase):
    async def test_search_sends_filters_and_lists_events(self):
        fake = self.use_fake(FakeDotNet(body=[{
            "name": "Summit", "location": "Beograd", "startDate": "2026-10-15T18:00:00",
            "endDate": "2026-10-15T21:00:00", "availablePlaces": 3, "capacity": 10,
        }]))
        tracker = FakeTracker(slots={
            "search_term": "Summit", "search_year": "2026", "search_month": "10",
            "only_available": True,
        })

        events = await self.run_action(actions.ActionSearchEvents, tracker)

        params = fake.requests[0].url.params
        self.assertEqual(params["searchTerm"], "Summit")
        self.assertEqual(params["startDate"], "2026-10-01")
        self.assertEqual(params["endDate"], "2026-10-31")
        self.assertEqual(params["availability"], "true")
        self.assertIn("Summit - Beograd", slot_value(events, "events_search_result"))
        self.assertEqual(slot_value(events, "last_listed_events"), ["Summit"])

    async def test_reservations_are_listed_with_ids(self):
        self.use_fake(FakeDotNet(body=[{
            "reservationId": 1008, "eventName": "Event Oct 2",
            "eventStartDate": "2026-10-26T10:00:00", "notes": None,
        }]))

        events = await self.run_action(actions.ActionListMyReservations, FakeTracker())

        self.assertIn("1008: Event Oct 2 (2026-10-26)", slot_value(events, "my_reservations_result"))

    async def test_unknown_event_availability_says_not_found(self):
        self.use_fake(FakeDotNet(status=404))

        events = await self.run_action(
            actions.ActionGetEventAvailability, FakeTracker(slots={"event_id": 999})
        )

        self.assertEqual(slot_value(events, "event_availability_result"), "Тражени догађај није пронађен.")

    async def test_reserve_posts_event_and_notes(self):
        fake = self.use_fake(FakeDotNet(body={"success": True}))
        tracker = FakeTracker(slots={"event_id": 5.0, "event_name": "Summit", "notes": "prvi red"})

        events = await self.run_action(actions.ActionReserveEvent, tracker)

        request = fake.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.content), {"eventId": 5, "notes": "prvi red"})
        self.assertIn("Summit", slot_value(events, "reservation_action_result"))

    async def test_reserve_already_reserved_code_is_reported_as_active(self):
        # Rezervacija postoji (ranije ili iz ponovljenog zahteva) - to nije greska za korisnika
        self.use_fake(FakeDotNet(body={
            "success": False, "errorCode": "AlreadyReserved",
            "errorMessage": "You already have a reservation for this event.",
        }))

        events = await self.run_action(
            actions.ActionReserveEvent, FakeTracker(slots={"event_id": 5, "event_name": "Summit"})
        )

        self.assertEqual(
            slot_value(events, "reservation_action_result"),
            "Већ имате резервацију за догађај Summit - резервација је активна.",
        )

    async def test_reserve_error_code_is_translated(self):
        self.use_fake(FakeDotNet(body={
            "success": False, "errorCode": "EventFull", "errorMessage": "Any text.",
        }))

        events = await self.run_action(
            actions.ActionReserveEvent, FakeTracker(slots={"event_id": 5, "event_name": "Summit"})
        )

        self.assertEqual(
            slot_value(events, "reservation_action_result"),
            "Догађај је попуњен и више не прима резервације.",
        )

    async def test_unknown_error_code_falls_back_to_generic_text(self):
        # Kod ima prednost: tekst se ne koristi ni kad bi ga mapa prepoznala
        self.use_fake(FakeDotNet(body={
            "success": False, "errorCode": "SomethingNew", "errorMessage": "Reservation not found.",
        }))

        events = await self.run_action(
            actions.ActionCancelReservation, FakeTracker(slots={"reservation_id": 1})
        )

        self.assertEqual(slot_value(events, "reservation_action_result"), "Отказивање резервације није успело.")

    async def test_reserve_error_without_code_is_translated_from_text(self):
        # Stariji .NET bez errorCode: poruka se prepoznaje po tekstu
        self.use_fake(FakeDotNet(body={
            "success": False, "errorMessage": "You already have a reservation for this event.",
        }))

        events = await self.run_action(
            actions.ActionReserveEvent, FakeTracker(slots={"event_id": 5, "event_name": "Summit"})
        )

        self.assertEqual(slot_value(events, "reservation_action_result"), "Већ имате резервацију за овај догађај.")

    async def test_cancel_sends_delete_for_the_reservation(self):
        fake = self.use_fake(FakeDotNet(body={"success": True, "reservationId": 1008}))

        events = await self.run_action(
            actions.ActionCancelReservation, FakeTracker(slots={"reservation_id": 1008.0})
        )

        self.assertEqual(fake.requests[0].method, "DELETE")
        self.assertEqual(fake.requests[0].url.path, "/api/internal/reservations/1008")
        self.assertEqual(slot_value(events, "reservation_action_result"), "Резервација је успешно отказана.")

    async def test_cancel_of_reservation_that_is_not_yours_explains_why(self):
        self.use_fake(FakeDotNet(body={
            "success": False, "errorCode": "ReservationNotFound", "errorMessage": "Reservation not found.",
        }))

        events = await self.run_action(
            actions.ActionCancelReservation, FakeTracker(slots={"reservation_id": 1005})
        )

        self.assertEqual(
            slot_value(events, "reservation_action_result"),
            "Резервација са тим ID-јем не постоји међу Вашим резервацијама.",
        )

    async def test_unknown_error_from_dotnet_falls_back_to_generic_text(self):
        self.use_fake(FakeDotNet(body={"success": False, "errorMessage": "Something new."}))

        events = await self.run_action(
            actions.ActionCancelReservation, FakeTracker(slots={"reservation_id": 1})
        )

        self.assertEqual(slot_value(events, "reservation_action_result"), "Отказивање резервације није успело.")


class FailureTests(ActionTestCase):
    async def assert_generic_error(self, fake: FakeDotNet) -> List[str]:
        self.use_fake(fake)
        with self.assertLogs("actions.actions", level="WARNING") as logs:
            events = await self.run_action(actions.ActionListMyReservations, FakeTracker())
        self.assertEqual(slot_value(events, "my_reservations_result"), actions.GENERIC_ERROR_MESSAGE)
        return logs.output

    async def test_server_error_gives_generic_message_and_no_body_in_log(self):
        output = "\n".join(await self.assert_generic_error(
            FakeDotNet(status=500, body={"notes": "PRIVATE-NOTE"})
        ))

        self.assertIn("GET /api/internal/reservations -> HTTP 500", output)
        self.assertNotIn("PRIVATE-NOTE", output)

    async def test_failure_log_carries_correlation_id(self):
        # Po njemu se ovaj log povezuje sa logovima .NET-a i Rase za istu poruku
        output = "\n".join(await self.assert_generic_error(
            FakeDotNet(error=httpx.ConnectError("connection refused"))
        ))

        self.assertIn(f"correlation_id={CORRELATION_ID}", output)

    async def test_failure_without_correlation_id_is_still_logged(self):
        self.use_fake(FakeDotNet(status=500))

        with self.assertLogs("actions.actions", level="WARNING") as logs:
            await self.run_action(
                actions.ActionListMyReservations, FakeTracker(correlation_id=None)
            )

        self.assertIn("correlation_id=-", "\n".join(logs.output))

    async def test_unreachable_dotnet_gives_generic_message(self):
        output = "\n".join(await self.assert_generic_error(
            FakeDotNet(error=httpx.ConnectError("connection refused"))
        ))

        self.assertIn("ConnectError", output)

    async def test_invalid_json_gives_generic_message(self):
        await self.assert_generic_error(FakeDotNet(raw_body=b"<html>not json</html>"))

    async def test_search_parameters_are_not_logged(self):
        # Pojam pretrage je tekst korisnika - ne sme u log ni kad poziv padne
        self.use_fake(FakeDotNet(status=500))
        tracker = FakeTracker(slots={"search_term": "Petar Petrovic", "location": "Beograd"})

        with self.assertLogs("actions.actions", level="WARNING") as logs:
            await self.run_action(actions.ActionSearchEvents, tracker)

        output = "\n".join(logs.output)
        self.assertIn("/api/internal/events/search", output)
        self.assertNotIn("Petar", output)
        self.assertNotIn("Beograd", output)


if __name__ == "__main__":
    unittest.main()
