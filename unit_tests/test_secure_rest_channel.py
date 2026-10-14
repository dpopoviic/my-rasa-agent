"""Testovi za secure_rest_channel.py - pre svega da uloge (user_roles) u metadata
poruke dolaze samo iz potpisanog tokena, nikad od klijenta.

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from secure_rest_channel import (
    USER_TOKEN_AUDIENCE,
    USER_TOKEN_ISSUER,
    USER_TOKEN_ROLES_CLAIM,
    SecureRestInput,
)

USER_ID = "user-123"
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


if __name__ == "__main__":
    unittest.main()
