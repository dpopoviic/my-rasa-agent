"""Testovi za secure_rest_channel.py - pre svega da uloge (user_roles) u metadata
poruke dolaze samo iz potpisanog tokena, nikad od klijenta, i da restart
razgovora prolazi iste provere kao poruka.

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import asyncio
import contextlib
import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import jwt
import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from rasa.shared.core.events import Restarted

from secure_rest_channel import (
    USER_TOKEN_AUDIENCE,
    USER_TOKEN_ISSUER,
    USER_TOKEN_ROLES_CLAIM,
    SecureRestInput,
)

USER_ID = "user-123"
MESSAGE_ID = "message-1"
CORRELATION_ID = "correlation-1"
NO_ROLES_CLAIM = object()


def new_private_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def sign_token(private_key, subject=USER_ID, roles=NO_ROLES_CLAIM) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "sub": subject,
        "iss": USER_TOKEN_ISSUER,
        "aud": USER_TOKEN_AUDIENCE,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=5),
        "jti": MESSAGE_ID,
        "cid": CORRELATION_ID,
    }
    if roles is not NO_ROLES_CLAIM:
        claims[USER_TOKEN_ROLES_CLAIM] = roles
    return jwt.encode(claims, private_key, algorithm="ES256")


def make_request(token, sender=USER_ID, metadata=None):
    body = {"sender": sender, "message": "zdravo"}
    if metadata is not None:
        body["metadata"] = metadata
    return SimpleNamespace(
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
        match_info={},
        json=body,
        ctx=SimpleNamespace(),
    )


def make_restart_request(token, sender=USER_ID):
    # POST /restart nosi samo sender u telu, bez poruke
    return SimpleNamespace(
        method="POST",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        match_info={},
        json={"sender": sender},
        ctx=SimpleNamespace(),
    )


class FakeAgent:
    """Belezi redosled poziva koje _restart_conversation pravi nad Rasa agentom."""

    def __init__(self):
        self.calls = []
        self.tracker = mock.Mock()
        self.tracker.update.side_effect = lambda event: self.calls.append(("update", event))
        agent = self

        class LockStore:
            @contextlib.asynccontextmanager
            async def lock(self, conversation_id):
                agent.calls.append(("lock", conversation_id))
                yield
                agent.calls.append(("unlock", conversation_id))

        class Processor:
            async def get_tracker(self, conversation_id):
                agent.calls.append(("get_tracker", conversation_id))
                return agent.tracker

            async def save_tracker(self, tracker):
                agent.calls.append(("save_tracker", tracker))

        self.lock_store = LockStore()
        self.processor = Processor()

    def cancel_background_tasks(self, sender_id):
        self.calls.append(("cancel", sender_id))
        return False


class SecureRestChannelRolesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.private_key = new_private_key()
        public_pem = cls.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        handle, cls.public_key_path = tempfile.mkstemp(suffix=".pem")
        with os.fdopen(handle, "wb") as f:
            f.write(public_pem)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.public_key_path)

    def setUp(self):
        with mock.patch.dict(os.environ, {"USER_TOKEN_PUBLIC_KEY_PATH": self.public_key_path}):
            self.channel = SecureRestInput()

    def authenticated_metadata(self, request):
        rejection = self.channel._authenticate(request)
        self.assertIsNone(rejection)
        return self.channel.get_metadata(request)

    def test_roles_from_token_reach_metadata(self):
        token = sign_token(self.private_key, roles=["Administrator", "Customer"])

        metadata = self.authenticated_metadata(make_request(token))

        self.assertEqual(metadata["user_roles"], ["Administrator", "Customer"])
        self.assertEqual(metadata["user_id"], USER_ID)

    def test_empty_roles_give_no_roles(self):
        token = sign_token(self.private_key, roles=[])

        metadata = self.authenticated_metadata(make_request(token))

        self.assertEqual(metadata["user_roles"], [])

    def test_token_without_roles_claim_gives_no_roles(self):
        # Token iz .NET aplikacije pre dodavanja uloga - i dalje prolazi, ali bez uloga
        token = sign_token(self.private_key)

        metadata = self.authenticated_metadata(make_request(token))

        self.assertEqual(metadata["user_roles"], [])

    def test_roles_claim_that_is_not_a_list_gives_no_roles(self):
        token = sign_token(self.private_key, roles="Administrator")

        metadata = self.authenticated_metadata(make_request(token))

        self.assertEqual(metadata["user_roles"], [])

    def test_non_string_roles_are_dropped(self):
        token = sign_token(self.private_key, roles=["Customer", "", 7, None])

        metadata = self.authenticated_metadata(make_request(token))

        self.assertEqual(metadata["user_roles"], ["Customer"])

    def test_roles_sent_by_client_are_ignored(self):
        token = sign_token(self.private_key, roles=["Customer"])
        request = make_request(
            token,
            metadata={"user_roles": ["Administrator"], "user_id": "victim", "language": "en"},
        )

        metadata = self.authenticated_metadata(request)

        self.assertEqual(metadata["user_roles"], ["Customer"])
        self.assertEqual(metadata["user_id"], USER_ID)
        self.assertEqual(metadata["language"], "en")

    def test_message_and_correlation_id_from_token_reach_metadata(self):
        metadata = self.authenticated_metadata(make_request(sign_token(self.private_key)))

        self.assertEqual(metadata["message_id"], MESSAGE_ID)
        self.assertEqual(metadata["correlation_id"], CORRELATION_ID)

    def test_message_and_correlation_id_sent_by_client_are_ignored(self):
        request = make_request(
            sign_token(self.private_key),
            metadata={"message_id": "forged", "correlation_id": "forged"},
        )

        metadata = self.authenticated_metadata(request)

        self.assertEqual(metadata["message_id"], MESSAGE_ID)
        self.assertEqual(metadata["correlation_id"], CORRELATION_ID)

    def test_correlation_id_is_bound_to_logs_and_cleared_for_next_request(self):
        self.authenticated_metadata(make_request(sign_token(self.private_key)))
        self.assertEqual(
            structlog.contextvars.get_contextvars().get("correlation_id"), CORRELATION_ID
        )

        # Sledeci zahtev na istoj vezi (odbijen) ne sme da nasledi tudji correlation_id
        self.channel._authenticate(make_request(sign_token(new_private_key())))
        self.assertNotIn("correlation_id", structlog.contextvars.get_contextvars())

    def test_token_signed_with_another_key_is_rejected(self):
        token = sign_token(new_private_key(), roles=["Administrator"])
        request = make_request(token)

        rejection = self.channel._authenticate(request)

        self.assertEqual(rejection.status, 401)
        self.assertFalse(hasattr(request.ctx, "user_roles"))

    def test_sender_other_than_token_subject_is_rejected(self):
        token = sign_token(self.private_key, roles=["Administrator"])
        request = make_request(token, sender="victim")

        rejection = self.channel._authenticate(request)

        self.assertEqual(rejection.status, 403)
        self.assertFalse(hasattr(request.ctx, "user_roles"))


class SecureRestChannelRestartTests(unittest.TestCase):
    """Restart (nova konverzacija) zamenjuje Rasin HTTP API sa AUTH_TOKEN-om, pa
    mora da prodje iste provere kao poruka: ispravan token i sender == sub."""

    @classmethod
    def setUpClass(cls):
        cls.private_key = new_private_key()
        public_pem = cls.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        handle, cls.public_key_path = tempfile.mkstemp(suffix=".pem")
        with os.fdopen(handle, "wb") as f:
            f.write(public_pem)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.public_key_path)

    def setUp(self):
        with mock.patch.dict(os.environ, {"USER_TOKEN_PUBLIC_KEY_PATH": self.public_key_path}):
            self.channel = SecureRestInput()

    def test_restart_for_own_conversation_is_authenticated(self):
        request = make_restart_request(sign_token(self.private_key))

        self.assertIsNone(self.channel._authenticate(request))
        self.assertEqual(request.ctx.user_id, USER_ID)

    def test_restart_without_token_is_rejected(self):
        rejection = self.channel._authenticate(make_restart_request(token=None))

        self.assertEqual(rejection.status, 401)

    def test_restart_with_token_signed_by_another_key_is_rejected(self):
        rejection = self.channel._authenticate(make_restart_request(sign_token(new_private_key())))

        self.assertEqual(rejection.status, 401)

    def test_restart_of_another_users_conversation_is_rejected(self):
        request = make_restart_request(sign_token(self.private_key), sender="victim")

        rejection = self.channel._authenticate(request)

        self.assertEqual(rejection.status, 403)
        self.assertFalse(hasattr(request.ctx, "user_id"))

    def test_restart_appends_restarted_and_saves_under_lock(self):
        agent = FakeAgent()

        asyncio.run(SecureRestInput._restart_conversation(agent, USER_ID))

        names = [name for name, _ in agent.calls]
        self.assertEqual(
            names, ["cancel", "lock", "get_tracker", "update", "save_tracker", "unlock"]
        )
        for name, arg in agent.calls:
            if name in ("cancel", "lock", "get_tracker", "unlock"):
                self.assertEqual(arg, USER_ID)
        restarted = next(arg for name, arg in agent.calls if name == "update")
        self.assertIsInstance(restarted, Restarted)


if __name__ == "__main__":
    unittest.main()
