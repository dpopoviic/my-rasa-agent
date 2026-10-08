"""Testovi za role_filtered_store.py - korisnik dobija samo delove dokumenata
koje njegove uloge smeju da citaju (pravila u knowledge_access.yml).

Pravi store je zamenjen sa FakeStore (ucitava se preko inner_type, isto kao u
Rasi), a pravila koriste prave nazive dokumenata iz Azure indeksa.

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import asyncio
import os
import tempfile
import time
import unicodedata
import unittest
from typing import Any, Dict, List

from rasa.core.information_retrieval import InformationRetrieval, SearchResult, SearchResultList
from rasa.utils.endpoints import EndpointConfig

from role_filtered_store import AccessRulesError, RoleFilteredStore

STUDENTS = "0006 Упутство за управљање подацима о ученицима.pdf"
SCHOOLS = "0005 Упутство за управљање подацима о школама.pdf"

RULES_YAML = f"""
match_field: title
admin_roles: [Administrator]
rules:
  - match: "{STUDENTS}"
    roles: [Customer]
"""


class FakeStore(InformationRetrieval):
    """Vraca unapred zadate rezultate i pamti kako je pozvan."""

    results: List[SearchResult] = []
    connected_with: List[EndpointConfig] = []
    search_calls = 0

    def connect(self, config: EndpointConfig) -> None:
        FakeStore.connected_with.append(config)

    async def search(self, query, tracker_state, threshold=0.0) -> SearchResultList:
        FakeStore.search_calls += 1
        return SearchResultList(results=list(FakeStore.results), metadata={"total_results": 99})


def chunk(title, text=None) -> SearchResult:
    metadata = {} if title is None else {"title": title}
    return SearchResult(text=text or f"deo iz {title}", metadata=metadata)


def tracker_state(roles) -> Dict[str, Any]:
    metadata = {} if roles is None else {"user_roles": roles}
    return {"latest_message": {"text": "pitanje", "metadata": metadata}}


class RoleFilteredStoreTests(unittest.TestCase):
    def setUp(self):
        FakeStore.results = [chunk(SCHOOLS, "skole 1"), chunk(STUDENTS, "ucenici 1"), chunk(SCHOOLS, "skole 2")]
        FakeStore.connected_with = []
        FakeStore.search_calls = 0
        handle, self.rules_path = tempfile.mkstemp(suffix=".yml")
        os.close(handle)
        self.write_rules(RULES_YAML)

    def tearDown(self):
        os.remove(self.rules_path)

    def write_rules(self, text):
        with open(self.rules_path, "w", encoding="utf-8") as f:
            f.write(text)

    def connected_store(self, **extra) -> RoleFilteredStore:
        store = RoleFilteredStore(embeddings=None)
        store.connect(self.config(**extra))
        return store

    def config(self, **extra) -> EndpointConfig:
        kwargs = {
            "type": "role_filtered_store.RoleFilteredStore",
            "inner_type": "unit_tests.test_role_filtered_store.FakeStore",
            "access_rules": self.rules_path,
            **extra,
        }
        return EndpointConfig(**kwargs)

    def search(self, store, roles) -> SearchResultList:
        return asyncio.run(store.search("pitanje", tracker_state(roles)))

    def texts(self, results: SearchResultList) -> List[str]:
        return [r.text for r in results.results]

    # --- ko sta vidi ---

    def test_customer_sees_only_students_document(self):
        results = self.search(self.connected_store(), ["Customer"])

        self.assertEqual(self.texts(results), ["ucenici 1"])

    def test_administrator_sees_everything(self):
        results = self.search(self.connected_store(), ["Administrator"])

        self.assertEqual(self.texts(results), ["skole 1", "ucenici 1", "skole 2"])

    def test_user_with_both_roles_sees_everything(self):
        results = self.search(self.connected_store(), ["Customer", "Administrator"])

        self.assertEqual(len(results.results), 3)

    def test_user_without_roles_sees_nothing_and_store_is_not_asked(self):
        store = self.connected_store()

        for roles in (None, [], "Administrator", [7, ""]):
            with self.subTest(roles=roles):
                self.assertEqual(self.search(store, roles).results, [])
        self.assertEqual(FakeStore.search_calls, 0)

    def test_unknown_role_sees_nothing(self):
        results = self.search(self.connected_store(), ["Guest"])

        self.assertEqual(results.results, [])

    def test_document_without_rule_is_hidden_from_non_admins(self):
        FakeStore.results = [chunk("0007 Novi dokument.pdf")]

        results = self.search(self.connected_store(), ["Customer"])

        self.assertEqual(results.results, [])

    def test_chunk_without_title_is_hidden_from_non_admins(self):
        FakeStore.results = [chunk(None), chunk(STUDENTS, "ucenici 1")]

        results = self.search(self.connected_store(), ["Customer"])

        self.assertEqual(self.texts(results), ["ucenici 1"])

    def test_title_in_other_unicode_form_still_matches(self):
        FakeStore.results = [chunk(unicodedata.normalize("NFD", STUDENTS), "ucenici 1")]

        results = self.search(self.connected_store(), ["Customer"])

        self.assertEqual(self.texts(results), ["ucenici 1"])

    def test_first_matching_rule_wins(self):
        self.write_rules(
            """
match_field: title
rules:
  - match: "0005 *"
    roles: [Administrator]
  - match: "*"
    roles: [Customer]
"""
        )

        results = self.search(self.connected_store(), ["Customer"])

        self.assertEqual(self.texts(results), ["ucenici 1"])

    def test_restricted_results_leave_no_trace_in_metadata(self):
        results = self.search(self.connected_store(), ["Customer"])

        self.assertEqual(results.metadata, {"total_results": 1})

    # --- broj rezultata i konfiguracija pravog store-a ---

    def test_max_results_is_applied_after_filtering(self):
        FakeStore.results = [chunk(SCHOOLS)] * 6 + [chunk(STUDENTS, f"ucenici {i}") for i in range(6)]

        results = self.search(self.connected_store(max_results=4), ["Customer"])

        self.assertEqual(self.texts(results), [f"ucenici {i}" for i in range(4)])

    def test_inner_store_gets_its_own_settings_without_wrapper_settings(self):
        self.connected_store(max_results=4, top_k=12, index_name="indeks")

        inner_config = FakeStore.connected_with[-1]
        self.assertEqual(inner_config.kwargs, {"top_k": 12, "index_name": "indeks"})
        self.assertEqual(inner_config.type, "unit_tests.test_role_filtered_store.FakeStore")

    def test_inner_store_is_created_once_and_connected_on_every_connect(self):
        store = self.connected_store()
        inner = store.inner

        store.connect(self.config())

        self.assertIs(store.inner, inner)
        self.assertEqual(len(FakeStore.connected_with), 2)

    # --- neispravna konfiguracija: pretraga ne radi umesto da radi bez pravila ---

    def test_missing_settings_fail_connect(self):
        for missing in ("inner_type", "access_rules"):
            with self.subTest(missing=missing):
                config = self.config()
                del config.kwargs[missing]
                with self.assertRaises(AccessRulesError):
                    RoleFilteredStore(embeddings=None).connect(config)

    def test_missing_rules_file_fails_connect(self):
        os.remove(self.rules_path)
        try:
            with self.assertRaises(AccessRulesError):
                self.connected_store()
        finally:
            self.write_rules(RULES_YAML)

    def test_invalid_rules_fail_connect(self):
        invalid = {
            "no match_field": "rules: []",
            "roles not a list": f'match_field: title\nrules:\n  - match: "x"\n    roles: Customer',
            "rule without match": "match_field: title\nrules:\n  - roles: [Customer]",
            "admin_roles not a list": "match_field: title\nadmin_roles: Administrator",
            "not a map": "- title",
        }
        for name, text in invalid.items():
            with self.subTest(name):
                self.write_rules(text)
                with self.assertRaises(AccessRulesError):
                    self.connected_store()

    def test_changed_rules_apply_without_restart(self):
        store = self.connected_store()
        self.assertEqual(len(self.search(store, ["Customer"]).results), 1)

        self.write_rules(RULES_YAML.replace(STUDENTS, SCHOOLS))
        bump_mtime(self.rules_path)
        store.connect(self.config())

        self.assertEqual(self.texts(self.search(store, ["Customer"])), ["skole 1", "skole 2"])

    def test_broken_rules_after_change_stop_search(self):
        store = self.connected_store()

        self.write_rules("rules: nije lista")
        bump_mtime(self.rules_path)
        with self.assertRaises(AccessRulesError):
            store.connect(self.config())

        # Stara pravila se ne koriste dalje
        with self.assertRaises(AccessRulesError):
            self.search(store, ["Customer"])


def bump_mtime(path):
    # Na nekim sistemima mtime ima grubu rezoluciju, pa se eksplicitno pomera
    later = time.time() + 5
    os.utime(path, (later, later))


if __name__ == "__main__":
    unittest.main()
