"""Baza znanja po ulogama: omotac oko bilo kog vector store-a.

EnterpriseSearchPolicy poziva ovu klasu umesto pravog store-a. Ona:
1. iz poslednje poruke uzima uloge korisnika (metadata.user_roles - upisuje ih
   secure_rest_channel.py iz potpisanog tokena, korisnik ih ne moze menjati);
2. pita pravi store (inner_type, npr. azure_search_store.AzureAISearch_Store);
3. od pronadjenih delova dokumenata zadrzava samo one koje te uloge smeju da
   citaju, po pravilima iz knowledge_access.yml.

Pravi store ne zna nista o ulogama, pa se moze zameniti (Qdrant, Milvus, ...)
bez izmene pravila. Sve sto nije jasno dozvoljeno se odbacuje: korisnik bez
uloga, deo bez polja iz match_field i dokument za koji nema pravila.
Ako ne ostane nista, polisa odgovara kao da nista nije pronadjeno
(utter_no_relevant_answer_found), pa korisnik ne saznaje da skriveni dokumenti postoje.

Konfiguracija (endpoints.yml, vector_store):
    type: role_filtered_store.RoleFilteredStore
    inner_type: azure_search_store.AzureAISearch_Store
    access_rules: knowledge_access.yml
    max_results: 4        # koliko delova najvise ide LLM-u posle filtriranja
    ...                   # sve ostalo (endpoint, index_name, top_k...) ide pravom store-u

Pravi store treba da vrati vise rezultata nego max_results (npr. top_k: 12),
da bi posle odbacivanja tudjih dokumenata ostalo dovoljno dozvoljenih.
"""

import copy
import fnmatch
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple

import structlog

from rasa.core.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
    SearchResult,
    SearchResultList,
    create_from_endpoint_config,
)
from rasa.shared.utils.yaml import read_yaml_file
from rasa.utils.endpoints import EndpointConfig

logger = structlog.get_logger()

DEFAULT_MAX_RESULTS = 4


class AccessRulesError(InformationRetrievalException):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        return self.base_message + self.message


def _normalize(text: Text) -> Text:
    # Isti naslov moze stici u razlicitim Unicode oblicima (npr. iz PDF-a ili YAML-a)
    return unicodedata.normalize("NFC", text)


def _string_list(value: Any, where: Text) -> List[Text]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise AccessRulesError(f"'{where}' mora biti lista nepraznih stringova.")
    return list(value)


@dataclass(frozen=True)
class AccessRule:
    pattern: Text  # * i ? kao u imenima fajlova (fnmatch)
    roles: frozenset


@dataclass(frozen=True)
class AccessRules:
    match_field: Text
    admin_roles: frozenset
    rules: Tuple[AccessRule, ...]

    @classmethod
    def from_dict(cls, data: Any) -> "AccessRules":
        if not isinstance(data, dict):
            raise AccessRulesError("Fajl sa pravilima mora biti YAML mapa.")

        match_field = data.get("match_field")
        if not isinstance(match_field, str) or not match_field:
            raise AccessRulesError("'match_field' je obavezan (npr. title).")

        admin_roles = _string_list(data.get("admin_roles", []), "admin_roles")

        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            raise AccessRulesError("'rules' mora biti lista.")
        rules = []
        for index, raw in enumerate(raw_rules):
            where = f"rules[{index}]"
            if not isinstance(raw, dict):
                raise AccessRulesError(f"'{where}' mora imati 'match' i 'roles'.")
            pattern = raw.get("match")
            if not isinstance(pattern, str) or not pattern:
                raise AccessRulesError(f"'{where}.match' je obavezan.")
            roles = _string_list(raw.get("roles"), f"{where}.roles")
            rules.append(AccessRule(_normalize(pattern), frozenset(roles)))

        return cls(match_field, frozenset(admin_roles), tuple(rules))

    def allows(self, metadata: Dict[Text, Any], roles: Sequence[Text]) -> bool:
        """Da li korisnik sa ovim ulogama sme da vidi deo dokumenta sa ovim metadata."""
        if self.admin_roles.intersection(roles):
            return True
        value = metadata.get(self.match_field)
        if not isinstance(value, str) or not value:
            return False
        value = _normalize(value)
        for rule in self.rules:
            # Prvo pravilo koje odgovara odlucuje; fnmatchcase razlikuje velika/mala slova
            if fnmatch.fnmatchcase(value, rule.pattern):
                return bool(rule.roles.intersection(roles))
        return False


def load_access_rules(path: Text) -> AccessRules:
    if not os.path.isfile(path):
        raise AccessRulesError(f"Fajl sa pravilima '{path}' ne postoji.")
    try:
        data = read_yaml_file(path, expand_env_vars=False, skip_cache=True)
    except Exception as e:
        raise AccessRulesError(f"Fajl sa pravilima '{path}' nije ispravan YAML: {e}") from e
    return AccessRules.from_dict(data)


def user_roles_from_tracker_state(tracker_state: Optional[Dict[Text, Any]]) -> List[Text]:
    """Uloge iz metadata poslednje poruke; bez ispravne liste - bez uloga."""
    latest_message = (tracker_state or {}).get("latest_message") or {}
    metadata = latest_message.get("metadata") or {}
    roles = metadata.get("user_roles") if isinstance(metadata, dict) else None
    if not isinstance(roles, list):
        return []
    return [role for role in roles if isinstance(role, str) and role]


class RoleFilteredStore(InformationRetrieval):
    def __init__(self, embeddings: Any) -> None:
        super().__init__(embeddings)
        self.inner: Optional[InformationRetrieval] = None
        self.inner_type: Optional[Text] = None
        self.rules: Optional[AccessRules] = None
        self._rules_source: Optional[Tuple[Text, float]] = None

    def connect(self, config: EndpointConfig) -> None:
        # Rasa poziva connect pre svake pretrage, ne samo pri startu
        params = dict(config.kwargs)
        inner_type = params.pop("inner_type", None)
        rules_path = params.pop("access_rules", None)
        if not inner_type:
            raise AccessRulesError("Nedostaje 'inner_type' u vector_store sekciji endpoints.yml.")
        if not rules_path:
            raise AccessRulesError("Nedostaje 'access_rules' u vector_store sekciji endpoints.yml.")
        self.max_results = int(params.pop("max_results", DEFAULT_MAX_RESULTS))

        self._load_rules_if_changed(str(rules_path))

        if self.inner is None or self.inner_type != inner_type:
            self.inner = create_from_endpoint_config(inner_type, self.embeddings)
            self.inner_type = inner_type
        inner_config = copy.copy(config)
        inner_config.type = inner_type
        inner_config.kwargs = params
        self.inner.connect(inner_config)

    def _load_rules_if_changed(self, path: Text) -> None:
        # Fajl se ponovo cita samo kad se promeni, pa izmena pravila vazi bez restarta Rase.
        # Neispravan fajl ne ostavlja stara pravila - pretraga pada dok se ne ispravi.
        try:
            source = (os.path.abspath(path), os.path.getmtime(path))
        except OSError:
            self.rules, self._rules_source = None, None
            raise AccessRulesError(f"Fajl sa pravilima '{path}' ne postoji.")
        if source == self._rules_source and self.rules is not None:
            return
        self.rules, self._rules_source = None, None
        self.rules = load_access_rules(path)
        self._rules_source = source
        logger.info("role_filtered_store.rules_loaded", path=path, rules=len(self.rules.rules))

    async def search(
        self, query: Text, tracker_state: Dict[str, Any], threshold: float = 0.0
    ) -> SearchResultList:
        if self.inner is None or self.rules is None:
            raise AccessRulesError("Store nije povezan (connect nije uspeo).")

        roles = user_roles_from_tracker_state(tracker_state)
        if not roles:
            # Npr. rasa shell ili poruka bez tokena: bez uloga nema dokumenata
            logger.info("role_filtered_store.no_roles")
            return SearchResultList(results=[], metadata={"total_results": 0})

        found = await self.inner.search(query, tracker_state, threshold)
        allowed: List[SearchResult] = [
            result for result in found.results if self.rules.allows(result.metadata, roles)
        ][: self.max_results]

        logger.debug(
            "role_filtered_store.search",
            roles=roles,
            found=len(found.results),
            allowed=len(allowed),
        )
        # Metadata pravog store-a se ne prenosi - mogla bi da otkrije koliko je skrivenih delova nadjeno
        return SearchResultList(results=allowed, metadata={"total_results": len(allowed)})
