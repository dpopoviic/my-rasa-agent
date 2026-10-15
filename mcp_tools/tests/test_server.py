"""Testovi MCP servera - .NET API je zamenjen laznim transportom, nista ne mora da radi.

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s mcp_tools/tests -t .
"""

import datetime
import json
import os
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_tools import server
from user_token import USER_TOKEN_AUDIENCE, USER_TOKEN_ISSUER


def new_private_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def public_pem(private_key: ec.EllipticCurvePrivateKey) -> str:
    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()


def sign_token(
    private_key: ec.EllipticCurvePrivateKey,
    subject: str = "user-123",
    audience: str = USER_TOKEN_AUDIENCE,
    expires_in: datetime.timedelta = datetime.timedelta(minutes=5),
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return jwt.encode(
        {"sub": subject, "iss": USER_TOKEN_ISSUER, "aud": audience,
         "iat": now, "exp": now + expires_in},
        private_key,
        algorithm="ES256",
    )


class FakeDotNet:
    """Lazni .NET interni API: pamti zahteve i vraca unapred zadat odgovor."""

    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status = status
        self.body = body
        self.requests: List[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)


class BuildSearchParamsTests(unittest.TestCase):
    def test_no_filters_returns_no_params(self) -> None:
        self.assertEqual(server.build_search_params(), {})

    def test_year_only_is_whole_year(self) -> None:
        self.assertEqual(
            server.build_search_params(year=2026),
            {"startDate": "2026-01-01", "endDate": "2026-12-31"},
        )

    def test_year_and_month_is_that_month(self) -> None:
        self.assertEqual(
            server.build_search_params(year=2026, month=2),
            {"startDate": "2026-02-01", "endDate": "2026-02-28"},
        )

    def test_explicit_dates_override_year_month_bounds(self) -> None:
        self.assertEqual(
            server.build_search_params(year=2026, month=10, start_date="2026-10-10"),
            {"startDate": "2026-10-10", "endDate": "2026-10-31"},
        )

    def test_only_start_date(self) -> None:
        self.assertEqual(
            server.build_search_params(start_date="2026-10-05"),
            {"startDate": "2026-10-05"},
        )

    def test_all_filters_combined(self) -> None:
        self.assertEqual(
            server.build_search_params(
                search_term=" Summit ", location="Beograd",
                start_date="2026-10-01", end_date="2026-10-15", only_available=True,
            ),
            {
                "searchTerm": "Summit", "location": "Beograd",
                "startDate": "2026-10-01", "endDate": "2026-10-15",
                "availability": "true",
            },
        )

    def test_empty_strings_and_false_mean_no_filter(self) -> None:
        self.assertEqual(
            server.build_search_params(search_term="  ", location="", only_available=False),
            {},
        )

    def test_month_without_year_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "requires `year`"):
            server.build_search_params(month=10)

    def test_invalid_month_is_rejected(self) -> None:
        with self.assertRaises(ToolError):
            server.build_search_params(year=2026, month=13)

    def test_invalid_date_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "YYYY-MM-DD"):
            server.build_search_params(start_date="10. oktobar")

    def test_start_after_end_is_rejected(self) -> None:
        with self.assertRaises(ToolError):
            server.build_search_params(start_date="2026-10-20", end_date="2026-10-01")


class ToolCallTests(unittest.IsolatedAsyncioTestCase):
    """Alati se pozivaju kroz pravu MCP sesiju (u memoriji), kao sto ce ih zvati Rasa."""

    def setUp(self) -> None:
        # MCP proverava token pre poziva .NET-u - testovi potpisuju prave tokene svojim kljucem
        self.private_key = new_private_key()
        original = server._user_token_public_key
        server._user_token_public_key = public_pem(self.private_key)
        self.addCleanup(setattr, server, "_user_token_public_key", original)
        self.token = sign_token(self.private_key)

    def use_fake(self, fake: FakeDotNet) -> None:
        original = server.api
        server.api = server.InternalApi(transport=httpx.MockTransport(fake.handler))
        self.addCleanup(setattr, server, "api", original)
        self.addAsyncCleanup(server.api.aclose)

    async def call(
        self, tool: str, args: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None
    ) -> Any:
        async with create_connected_server_and_client_session(server.mcp._mcp_server) as session:
            return await session.call_tool(tool, args or {}, meta=meta)

    async def test_user_token_from_meta_is_forwarded_as_bearer(self) -> None:
        fake = FakeDotNet(body=[])
        self.use_fake(fake)

        result = await self.call("search_events", {"year": 2026, "month": 10}, {"user_token": self.token})

        self.assertFalse(result.isError)
        request = fake.requests[0]
        self.assertEqual(request.headers["Authorization"], f"Bearer {self.token}")
        self.assertIn("X-Internal-Api-Key", request.headers)
        self.assertEqual(request.url.path, "/api/internal/events/search")
        self.assertEqual(request.url.params["startDate"], "2026-10-01")
        self.assertEqual(request.url.params["endDate"], "2026-10-31")

    async def test_missing_token_fails_without_calling_dotnet(self) -> None:
        fake = FakeDotNet(body=[])
        self.use_fake(fake)

        result = await self.call("get_my_reservations")

        self.assertTrue(result.isError)
        self.assertIn("not signed in", result.content[0].text)
        self.assertEqual(fake.requests, [])

    async def test_search_returns_compact_events_with_full_times(self) -> None:
        self.use_fake(FakeDotNet(body=[{
            "eventId": 3, "name": "Самит", "location": "Beograd",
            "startDate": "2026-10-15T18:00:00", "endDate": "2026-10-15T21:30:00",
            "capacity": 10, "reservedCount": 4, "availablePlaces": 6, "isFull": False,
            "currentUserHasReservation": True, "description": "x" * 500,
        }]))

        result = await self.call("search_events", {}, {"user_token": self.token})

        data = json.loads(result.structuredContent["result"])
        self.assertEqual(data["count"], 1)
        self.assertFalse(data["truncated"])
        event = data["events"][0]
        self.assertEqual(event["name"], "Самит")  # naziv nepromenjen, cirilica citljiva
        self.assertEqual(event["start"], "2026-10-15T18:00:00")
        self.assertEqual(event["end"], "2026-10-15T21:30:00")
        self.assertTrue(event["user_has_reservation"])
        self.assertLessEqual(len(event["description"]), server.MAX_DESCRIPTION_LENGTH + 3)

    async def test_search_caps_number_of_events(self) -> None:
        events = [{"eventId": i, "name": f"E{i}"} for i in range(server.MAX_EVENTS + 5)]
        self.use_fake(FakeDotNet(body=events))

        result = await self.call("search_events", {}, {"user_token": self.token})

        data = json.loads(result.structuredContent["result"])
        self.assertEqual(data["count"], server.MAX_EVENTS + 5)
        self.assertTrue(data["truncated"])
        self.assertEqual(len(data["events"]), server.MAX_EVENTS)

    async def test_month_without_year_is_an_error_for_the_llm(self) -> None:
        fake = FakeDotNet(body=[])
        self.use_fake(fake)

        result = await self.call("search_events", {"month": 10}, {"user_token": self.token})

        self.assertTrue(result.isError)
        self.assertIn("requires `year`", result.content[0].text)
        self.assertEqual(fake.requests, [])

    async def test_my_reservations_include_end_and_location(self) -> None:
        self.use_fake(FakeDotNet(body=[{
            "reservationId": 7, "eventId": 3, "eventName": "Summit",
            "eventStartDate": "2026-10-15T18:00:00", "eventEndDate": "2026-10-15T21:30:00",
            "eventLocation": "Beograd", "notes": None,
        }]))

        result = await self.call("get_my_reservations", {}, {"user_token": self.token})

        reservation = json.loads(result.structuredContent["result"])["reservations"][0]
        self.assertEqual(reservation["end"], "2026-10-15T21:30:00")
        self.assertEqual(reservation["location"], "Beograd")

    async def test_unknown_event_availability_is_an_error(self) -> None:
        self.use_fake(FakeDotNet(status=404))

        result = await self.call("get_event_availability", {"event_id": 999}, {"user_token": self.token})

        self.assertTrue(result.isError)
        self.assertIn("999", result.content[0].text)

    async def test_calls_reuse_one_http_client(self) -> None:
        # Connection pooling: isti klijent (i njegove veze) za sve pozive alata
        self.use_fake(FakeDotNet(body=[]))

        await self.call("get_my_reservations", {}, {"user_token": self.token})
        first_client = server.api._client
        await self.call("search_events", {}, {"user_token": self.token})

        self.assertIsNotNone(first_client)
        self.assertIs(server.api._client, first_client)

    async def test_error_log_has_no_response_body(self) -> None:
        # Telo odgovora moze da sadrzi podatke korisnika - ne sme u log
        self.use_fake(FakeDotNet(status=500, body={"notes": "PRIVATE-NOTE"}))

        with self.assertLogs("mcp_tools", level="WARNING") as logs:
            result = await self.call("get_my_reservations", {}, {"user_token": self.token})

        self.assertTrue(result.isError)
        output = "\n".join(logs.output)
        self.assertIn("/api/internal/reservations", output)
        self.assertIn("500", output)
        self.assertNotIn("PRIVATE-NOTE", output)

    async def test_token_rejected_by_dotnet_is_an_error(self) -> None:
        # Token je ispravan za MCP, ali ga .NET odbija (npr. nalog zakljucan)
        self.use_fake(FakeDotNet(status=401))

        result = await self.call("get_my_reservations", {}, {"user_token": self.token})

        self.assertTrue(result.isError)
        self.assertIn("not authorized", result.content[0].text)

    # --- Rana provera: los token se odbija u MCP-u, .NET se ne zove ---

    async def assert_rejected_without_calling_dotnet(self, token: str) -> None:
        fake = FakeDotNet(body=[])
        self.use_fake(fake)

        result = await self.call("get_my_reservations", {}, {"user_token": token})

        self.assertTrue(result.isError)
        self.assertIn("session is not valid", result.content[0].text)
        self.assertEqual(fake.requests, [])

    async def test_expired_token_is_rejected_without_calling_dotnet(self) -> None:
        expired = sign_token(self.private_key, expires_in=datetime.timedelta(minutes=-5))
        await self.assert_rejected_without_calling_dotnet(expired)

    async def test_token_signed_with_another_key_is_rejected_without_calling_dotnet(self) -> None:
        await self.assert_rejected_without_calling_dotnet(sign_token(new_private_key()))

    async def test_token_for_another_audience_is_rejected_without_calling_dotnet(self) -> None:
        other_audience = sign_token(self.private_key, audience="some-other-api")
        await self.assert_rejected_without_calling_dotnet(other_audience)

    async def test_malformed_token_is_rejected_without_calling_dotnet(self) -> None:
        await self.assert_rejected_without_calling_dotnet("not-a-jwt")


class StartupTests(unittest.TestCase):
    def test_missing_public_key_fails_at_startup(self) -> None:
        # Bez javnog kljuca MCP ne sme da se podigne (inace bi svaki poziv pao tek u radu)
        missing = os.path.join(os.path.dirname(__file__), "no-such-key.pem")
        with mock.patch.object(server, "MCP_API_KEY", "secret"), \
                mock.patch.object(server, "_user_token_public_key", None), \
                mock.patch.dict(os.environ, {"USER_TOKEN_PUBLIC_KEY_PATH": missing}):
            with self.assertRaises(FileNotFoundError):
                server.create_app()


class ApiKeyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        async def ok_app(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        app = server.require_api_key(ok_app, "secret")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_missing_key_is_rejected(self) -> None:
        self.assertEqual((await self.client.post("/mcp")).status_code, 401)

    async def test_wrong_key_is_rejected(self) -> None:
        response = await self.client.post("/mcp", headers={"X-Mcp-Api-Key": "wrong"})
        self.assertEqual(response.status_code, 401)

    async def test_correct_key_is_accepted(self) -> None:
        response = await self.client.post("/mcp", headers={"X-Mcp-Api-Key": "secret"})
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
