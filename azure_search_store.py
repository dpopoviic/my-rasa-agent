import hashlib
import os
import re
from typing import Any, Dict, List, Text

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

    async def search(
        self, query: Text, tracker_state: Dict[str, Any], threshold: float = 0.0
    ) -> SearchResultList:
        body: Dict[str, Any] = {
            "search": query,
            "top": self.top_k,
            "select": ",".join([self.content_field, *self.metadata_fields]),
        }
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

        url = (
            f"{self.endpoint}/indexes/{self.index_name}/docs/search"
            f"?api-version={self.api_version}"
        )
        logger.debug("azure_search.search", query=query, index=self.index_name)
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
