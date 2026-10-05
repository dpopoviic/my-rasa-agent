"""REST kanal koji prima poruke samo od .NET aplikacije, u ime tacno jednog korisnika.

Zamenjuje ugradjeni `rest` kanal (credentials.yml). Svaki zahtev mora da ima
`Authorization: Bearer <token>`, gde je token JWT koji je .NET aplikacija
(UserTokenService) potpisala svojim privatnim EC kljucem za ulogovanog korisnika.
Ovde se proverava samo javnim kljucem, pa Rasa moze da proveri token, ali ne
moze sama da ga napravi za drugog korisnika.

Provere:
1. potpis, izdavac (iss), namena (aud) i rok (exp) tokena;
2. `sub` iz tokena mora biti isti kao `sender` poruke (ili sender_id u putanji),
   pa se ne moze poslati poruka u tudji razgovor.

Metadata poruke pravi kanal sam, iz proverenog tokena ({user_id, user_token}).
Od onoga sto je klijent poslao u "metadata" prihvata se samo "language", i to
samo jezik iz config.yml - Rasa ga na pocetku sesije upisuje u slot `language`.
Akcije (actions/actions.py) prosledjuju user_token .NET internom API-ju, koji
ga ponovo proverava.
"""

import os
from typing import Any, Dict, Optional, Text

import jwt
import structlog
from sanic import Blueprint, response
from sanic.request import Request

from rasa.core.channels.channel import OnNewMessageType
from rasa.core.channels.rest import RestInput
from rasa.shared.utils.yaml import read_yaml_file

logger = structlog.get_logger()

# Moraju biti iste kao Issuer / Audience u .NET UserTokenService
USER_TOKEN_ISSUER = "event-reservation-app"
USER_TOKEN_AUDIENCE = "event-reservation-chatbot"
USER_TOKEN_ALGORITHMS = ["ES256"]
CLOCK_SKEW_SECONDS = 30

BEARER_PREFIX = "Bearer "


def _load_public_key() -> Text:
    path = os.environ.get("USER_TOKEN_PUBLIC_KEY_PATH", "keys/user_token_public.pem")
    with open(path, encoding="utf-8") as f:
        return f.read()


class SecureRestInput(RestInput):
    @classmethod
    def name(cls) -> Text:
        # Webhook: POST /webhooks/secure_rest/webhook
        return "secure_rest"

    def __init__(self) -> None:
        super().__init__()
        # Ucitava se pri startu - bez javnog kljuca Rasa ne treba ni da se podigne
        self.public_key = _load_public_key()
        config = read_yaml_file("config.yml")
        self.languages = {config["language"], *(config.get("additional_languages") or [])}

    def _verify_token(self, token: Text) -> Optional[Text]:
        """Vraca user id (sub) iz ispravnog tokena, inace None."""
        try:
            claims = jwt.decode(
                token,
                self.public_key,
                algorithms=USER_TOKEN_ALGORITHMS,
                audience=USER_TOKEN_AUDIENCE,
                issuer=USER_TOKEN_ISSUER,
                leeway=CLOCK_SKEW_SECONDS,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as e:
            logger.warning("secure_rest.invalid_token", error=str(e))
            return None
        return claims.get("sub") or None

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        # Provereni podaci iz middleware-a; iz tela zahteva samo poznat jezik
        metadata = {
            "user_id": request.ctx.user_id,
            "user_token": request.ctx.user_token,
        }
        client_metadata = (request.json or {}).get("metadata") or {}
        language = client_metadata.get("language") if isinstance(client_metadata, dict) else None
        if isinstance(language, str) and language in self.languages:
            metadata["language"] = language
        return metadata

    def _extract_input_channel(self, req: Request) -> Text:
        return self.name()

    def blueprint(self, on_new_message: OnNewMessageType) -> Blueprint:
        custom_webhook = super().blueprint(on_new_message)

        @custom_webhook.middleware("request")
        async def authenticate(request: Request) -> Optional[response.HTTPResponse]:
            if request.method == "GET":
                return None  # health check, ne prima poruke

            header = request.headers.get("Authorization") or ""
            if not header.startswith(BEARER_PREFIX):
                return response.json({"error": "missing user token"}, status=401)

            token = header[len(BEARER_PREFIX):].strip()
            user_id = self._verify_token(token)
            if user_id is None:
                return response.json({"error": "invalid user token"}, status=401)

            # /webhook ima sender u telu, /cancel_background_tasks/<sender_id> u putanji
            sender_id = request.match_info.get("sender_id")
            if sender_id is None:
                body = request.json if isinstance(request.json, dict) else {}
                sender_id = body.get("sender")

            if sender_id != user_id:
                logger.warning("secure_rest.sender_mismatch", sender_id=sender_id)
                return response.json({"error": "sender does not match token"}, status=403)

            request.ctx.user_id = user_id
            request.ctx.user_token = token
            return None

        return custom_webhook
