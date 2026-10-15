"""Testovi za user_token.py - jedino mesto sa pravilima provere tokena korisnika,
koje koriste i secure_rest kanal i MCP server. Token prolazi samo ako ga je
potpisala .NET aplikacija (ES256, njen kljuc), za ovu namenu i dok jos vazi.

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import base64
import datetime
import hashlib
import hmac
import json
import unittest

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from user_token import (
    CLOCK_SKEW_SECONDS,
    USER_TOKEN_AUDIENCE,
    USER_TOKEN_ISSUER,
    decode_user_token,
)


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class DecodeUserTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.private_key = ec.generate_private_key(ec.SECP256R1())
        cls.public_key = cls.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

    def claims(self, **overrides):
        now = datetime.datetime.now(datetime.timezone.utc)
        claims = {
            "sub": "user-123",
            "iss": USER_TOKEN_ISSUER,
            "aud": USER_TOKEN_AUDIENCE,
            "iat": now,
            "exp": now + datetime.timedelta(minutes=5),
        }
        claims.update(overrides)
        return {k: v for k, v in claims.items() if v is not None}

    def sign(self, claims, key=None):
        return jwt.encode(claims, key or self.private_key, algorithm="ES256")

    def assert_rejected(self, token):
        with self.assertRaises(jwt.PyJWTError):
            decode_user_token(token, self.public_key)

    def test_valid_token_returns_claims(self):
        claims = decode_user_token(self.sign(self.claims()), self.public_key)

        self.assertEqual(claims["sub"], "user-123")

    def test_token_signed_with_another_key_is_rejected(self):
        self.assert_rejected(self.sign(self.claims(), key=ec.generate_private_key(ec.SECP256R1())))

    def test_expired_token_is_rejected(self):
        expired = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=2)
        self.assert_rejected(self.sign(self.claims(exp=expired)))

    def test_token_expired_within_clock_skew_is_accepted(self):
        # Satovi servera se razlikuju za nekoliko sekundi - to ne sme da obori ispravan token
        just_expired = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            seconds=CLOCK_SKEW_SECONDS // 2
        )
        decode_user_token(self.sign(self.claims(exp=just_expired)), self.public_key)

    def test_wrong_issuer_is_rejected(self):
        self.assert_rejected(self.sign(self.claims(iss="some-other-app")))

    def test_wrong_audience_is_rejected(self):
        self.assert_rejected(self.sign(self.claims(aud="some-other-api")))

    def test_missing_expiration_is_rejected(self):
        self.assert_rejected(self.sign(self.claims(exp=None)))

    def test_missing_subject_is_rejected(self):
        self.assert_rejected(self.sign(self.claims(sub=None)))

    def test_empty_subject_is_rejected(self):
        self.assert_rejected(self.sign(self.claims(sub="")))

    def test_unsigned_token_is_rejected(self):
        # "alg": "none" - token bez potpisa
        header = b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        payload = b64url(json.dumps({"sub": "victim", "iss": USER_TOKEN_ISSUER,
                                     "aud": USER_TOKEN_AUDIENCE, "exp": 9999999999}).encode())
        self.assert_rejected(f"{header}.{payload}.")

    def test_hs256_token_signed_with_public_key_is_rejected(self):
        # Klasican napad: javni kljuc (koji je javan) iskoriscen kao HMAC tajna uz alg HS256
        header = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = b64url(json.dumps({"sub": "victim", "iss": USER_TOKEN_ISSUER,
                                     "aud": USER_TOKEN_AUDIENCE, "exp": 9999999999}).encode())
        signature = hmac.new(self.public_key.encode(), f"{header}.{payload}".encode(),
                             hashlib.sha256).digest()
        self.assert_rejected(f"{header}.{payload}.{b64url(signature)}")

    def test_malformed_token_is_rejected(self):
        self.assert_rejected("not-a-jwt")


if __name__ == "__main__":
    unittest.main()
