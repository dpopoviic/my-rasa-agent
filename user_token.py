"""Provera korisnickog tokena koji potpisuje .NET UserTokenService.

Jedno mesto za pravila provere (javni kljuc, iss, aud, algoritam, rok, obavezni
claim-ovi). Koriste ga:
- secure_rest kanal (secure_rest_channel.py) - na ulazu poruke u Rasu;
- MCP server (mcp_tools/server.py) - pre nego sto token prosledi .NET-u.

Namerno ne uvozi Rasu: MCP server je poseban, lagan proces i ne treba da ucitava
celu Rasu zbog jedne provere. Kad se predje na JWKS i rotaciju kljuceva, menja se
samo ovaj modul.
"""

import os
from typing import Any, Dict, Text

import jwt

# Moraju biti iste kao Issuer / Audience / *Claim konstante u .NET UserTokenService
USER_TOKEN_ISSUER = "event-reservation-app"
USER_TOKEN_AUDIENCE = "event-reservation-chatbot"
USER_TOKEN_ROLES_CLAIM = "roles"
# ID poruke (nov za svaki token) i correlation id (prati obradu jedne poruke kroz
# Rasu, akcije, MCP i .NET). Potpisani su, pa ih ni LLM ni korisnik ne mogu promeniti.
USER_TOKEN_MESSAGE_ID_CLAIM = "jti"
USER_TOKEN_CORRELATION_ID_CLAIM = "cid"
# Samo ES256: token sa "alg": "none" ili HS256 (javni kljuc kao HMAC tajna) se odbija
USER_TOKEN_ALGORITHMS = ["ES256"]
CLOCK_SKEW_SECONDS = 30

PUBLIC_KEY_PATH_ENV = "USER_TOKEN_PUBLIC_KEY_PATH"
DEFAULT_PUBLIC_KEY_PATH = "keys/user_token_public.pem"


def load_public_key() -> Text:
    """Javni kljuc iz USER_TOKEN_PUBLIC_KEY_PATH; bez njega servis ne treba ni da se podigne."""
    path = os.environ.get(PUBLIC_KEY_PATH_ENV, DEFAULT_PUBLIC_KEY_PATH)
    with open(path, encoding="utf-8") as f:
        return f.read()


def decode_user_token(token: Text, public_key: Text) -> Dict[Text, Any]:
    """Claim-ovi ispravnog tokena.

    Baca jwt.PyJWTError ako token nije ispravan: los potpis, drugi algoritam,
    pogresan iss/aud, istekao, nedostaje obavezni claim ili je `sub`, `jti` ili
    `cid` prazan.
    """
    claims = jwt.decode(
        token,
        public_key,
        algorithms=USER_TOKEN_ALGORITHMS,
        audience=USER_TOKEN_AUDIENCE,
        issuer=USER_TOKEN_ISSUER,
        leeway=CLOCK_SKEW_SECONDS,
        options={"require": [
            "exp", "iss", "aud", "sub",
            USER_TOKEN_MESSAGE_ID_CLAIM, USER_TOKEN_CORRELATION_ID_CLAIM,
        ]},
    )
    for name in ("sub", USER_TOKEN_MESSAGE_ID_CLAIM, USER_TOKEN_CORRELATION_ID_CLAIM):
        value = claims.get(name)
        if not isinstance(value, str) or not value:
            raise jwt.InvalidTokenError(f"Token has an empty or invalid '{name}' claim")
    return claims
