"""Baza znanja u Azure AI Search indeksu, sa filtriranjem po ulogama korisnika.

Svaki deo dokumenta (chunk) u indeksu ima polje `filter_field` (access_group) sa
grupom kojoj dokument pripada - upisuje ga indexer iz metadata PDF-a u Blob
Storage-u (vidi azure/skillset.json). Uloge korisnika dolaze iz metadata poslednje
poruke (user_roles - upisuje ih secure_rest_channel.py iz potpisanog tokena).
role_groups (endpoints.yml) kaze koje grupe koja uloga sme da cita, a Azure vraca
samo delove iz tih grupa - filter se primenjuje pre rangiranja, pa korisnik dobija
najbolje DOZVOLJENE delove.

Sve sto nije jasno dozvoljeno se odbija: korisnik bez uloga ili samo sa nepoznatim
ulogama ne dobija nista (Azure se i ne poziva), deo bez grupe vidi samo uloga sa "*".
"""

import hashlib
import os
import re
from typing import Any, Dict, FrozenSet, List, Optional, Text, Tuple

import httpx
import structlog

from rasa.core.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
    SearchResultList,
)
from rasa.core.information_retrieval.models import Document
from rasa.shared.providers.embedding.embedding_utils import aembed_query
from rasa.utils.endpoints import EndpointConfig

logger = structlog.get_logger()

DEFAULT_API_VERSION = "2024-07-01"
DEFAULT_CONTENT_FIELD = "content"
DEFAULT_TOP_K = 4
DEFAULT_TIMEOUT = 10


_ENV_REF = re.compile(r"^\$\{(\w+)\}$")

# "*" u role_groups: uloga vidi sve dokumente (pretraga bez filtera)
ALL_GROUPS = "*"
# Grupe ulaze u OData filter - dozvoljeni su samo ovi znakovi, pa nema potrebe za escape-ovanjem
_GROUP_NAME = re.compile(r"^[a-z0-9_-]+$")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_env(value: str) -> str:
    """Vraca vrednost promenljive okruzenja ako je value oblika ${NAZIV}."""
    match = _ENV_REF.match(value.strip())
    if not match:
        return value
    resolved = os.environ.get(match.group(1))
    if not resolved:
        raise AzureSearchException(
            f"Promenljiva okruzenja '{match.group(1)}' nije podesena (proveri .env)."
        )
    return resolved


class AzureSearchException(InformationRetrievalException):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        return self.base_message + self.message


def _parse_role_groups(value: Any) -> Dict[Text, FrozenSet[Text]]:
    """role_groups iz endpoints.yml: {uloga: [grupa, ...]}; svaka greska zaustavlja start."""
    if not isinstance(value, dict) or not value:
        raise AzureSearchException(
            "'role_groups' mora biti mapa uloga na liste grupa, npr. Customer: [public, customer]."
        )
    role_groups = {}
    for role, groups in value.items():
        if not isinstance(role, str) or not role:
            raise AzureSearchException(f"Neispravna uloga u role_groups: {role!r}.")
        if not isinstance(groups, list) or not groups:
            raise AzureSearchException(f"role_groups.{role} mora biti neprazna lista grupa.")
        for group in groups:
            if group != ALL_GROUPS and not (isinstance(group, str) and _GROUP_NAME.match(group)):
                raise AzureSearchException(
                    f"Neispravna grupa {group!r} za ulogu {role}: dozvoljena su mala slova, cifre, - i _ (ili \"*\")."
                )
        role_groups[role] = frozenset(groups)
    return role_groups


def user_roles_from_tracker_state(tracker_state: Optional[Dict[Text, Any]]) -> List[Text]:
    """Uloge iz metadata poslednje poruke; bez ispravne liste - bez uloga."""
    latest_message = (tracker_state or {}).get("latest_message") or {}
    metadata = latest_message.get("metadata") or {}
    roles = metadata.get("user_roles") if isinstance(metadata, dict) else None
    if not isinstance(roles, list):
        return []
    return [role for role in roles if isinstance(role, str) and role]


class AzureAISearch_Store(InformationRetrieval):
    def connect(self, config: EndpointConfig) -> None:
        params = config.kwargs
        for required in ("endpoint", "index_name", "api_key"):
            if not params.get(required):
                raise AzureSearchException(
                    f"Nedostaje '{required}' u vector_store sekciji endpoints.yml."
                )
        self.endpoint = str(params["endpoint"]).rstrip("/")
        self.index_name = str(params["index_name"])
        # Rasa ne zamenjuje ${VAR} u polju `api_key` (cuva ga kao tajnu), pa se
        # ovde razresava iz okruzenja - inace bi se poslao doslovni tekst "${...}".
        self.api_key = _resolve_env(str(params["api_key"]))
        self.api_version = str(params.get("api_version", DEFAULT_API_VERSION))
        self.content_field = params.get("content_field", DEFAULT_CONTENT_FIELD)
        # Ostala polja koja se vracaju kao metadata (npr. naslov, izvor)
        self.metadata_fields: List[str] = list(params.get("metadata_fields") or [])
        self.top_k = int(params.get("top_k", DEFAULT_TOP_K))
        self.timeout = int(params.get("timeout", DEFAULT_TIMEOUT))
        # Ako je zadato, koristi se semanticka rangiranje (mora postojati u indeksu)
        self.semantic_configuration = params.get("semantic_configuration")
        # Ako je zadato, radi se i vektorska pretraga (hibridno) preko embeddings modela
        self.vector_field = params.get("vector_field")
        # Filtriranje po ulogama je obavezno - bez ovoga store ne radi (ne pretrazuje sve)
        filter_field = params.get("filter_field")
        if not isinstance(filter_field, str) or not _FIELD_NAME.match(filter_field):
            raise AzureSearchException(
                "Nedostaje ili je neispravno 'filter_field' u vector_store sekciji endpoints.yml (npr. access_group)."
            )
        self.filter_field = filter_field
        self.role_groups = _parse_role_groups(params.get("role_groups"))

    def _groups_for(self, roles: List[Text]) -> Tuple[bool, FrozenSet[Text]]:
        """(vidi_sve, grupe) za uloge korisnika - unija grupa svih njegovih uloga."""
        unknown = [role for role in roles if role not in self.role_groups]
        if unknown:
            # Uloga postoji u .NET-u, a nije dodata u role_groups - ne daje nista
            logger.warning("azure_search.unknown_roles", roles=unknown)
        groups = frozenset().union(*(self.role_groups.get(role, frozenset()) for role in roles))
        if ALL_GROUPS in groups:
            return True, frozenset()
        return False, groups

    def _role_filter(self, groups: FrozenSet[Text]) -> Text:
        # search.in je "ili": deo prolazi ako mu je grupa bilo koja od navedenih
        return f"search.in({self.filter_field}, '{','.join(sorted(groups))}', ',')"

    async def search(
        self, query: Text, tracker_state: Dict[str, Any], threshold: float = 0.0
    ) -> SearchResultList:
        roles = user_roles_from_tracker_state(tracker_state)
        sees_all, groups = self._groups_for(roles)
        if not sees_all and not groups:
            # Bez uloga (npr. rasa shell) ili samo nepoznate uloge: nema dokumenata, Azure se ne poziva
            logger.info("azure_search.no_allowed_groups", roles=roles)
            return SearchResultList.from_document_list([])

        body: Dict[str, Any] = {
            "search": query,
            "top": self.top_k,
            "select": ",".join([self.content_field, *self.metadata_fields]),
        }
        if not sees_all:
            body["filter"] = self._role_filter(groups)
        if self.semantic_configuration:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = self.semantic_configuration
        if self.vector_field:
            embedding = await aembed_query(self.embeddings, query)
            body["vectorQueries"] = [
                {
                    "kind": "vector",
                    "vector": embedding,
                    "fields": self.vector_field,
                    "k": self.top_k,
                }
            ]
            if "filter" in body:
                # Filter pre izbora najblizih vektora: k najboljih DOZVOLJENIH delova
                # (podrazumevano i ovako, ali eksplicitno da ne zavisi od podesavanja indeksa)
                body["vectorFilterMode"] = "preFilter"

        url = (
            f"{self.endpoint}/indexes/{self.index_name}/docs/search"
            f"?api-version={self.api_version}"
        )
        logger.debug(
            "azure_search.search",
            query=query,
            index=self.index_name,
            roles=roles,
            filter=body.get("filter", "none (sees all)"),
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url, json=body, headers={"api-key": self.api_key}
                )
                response.raise_for_status()
        except Exception as e:
            # Dijagnostika: telo odgovora kaze zasto je Azure odbio zahtev, a otisak
            # kljuca (duzina + prvih 6 znakova SHA-256) pokazuje koji kljuc je Rasa
            # zaista ucitala, bez otkrivanja samog kljuca.
            body_text = ""
            if isinstance(e, httpx.HTTPStatusError):
                body_text = f" | odgovor: {e.response.text[:300]}"
            fingerprint = hashlib.sha256(self.api_key.encode()).hexdigest()[:6]
            raise AzureSearchException(
                f"Pretraga Azure AI Search indeksa nije uspela: {e}{body_text}"
                f" | kljuc: duzina={len(self.api_key)}, sha256={fingerprint}"
                f" | index={self.index_name}"
            ) from e

        documents: List[Document] = []
        for hit in response.json().get("value", []):
            text = hit.get(self.content_field)
            if not text:
                continue
            metadata = {f: hit[f] for f in self.metadata_fields if f in hit}
            documents.append(Document(text=str(text), metadata=metadata))
        return SearchResultList.from_document_list(documents)
