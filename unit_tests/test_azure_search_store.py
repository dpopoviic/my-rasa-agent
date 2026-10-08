"""Testovi za filtriranje po ulogama u azure_search_store.py - sta tacno ide Azure-u.

HTTP poziv je zamenjen laznim klijentom koji pamti telo zahteva, pa se proverava
filter koji Rasa salje (Azure ga onda primenjuje pre rangiranja).

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import asyncio
import unittest
from typing import Any, Dict, List
from unittest import mock

from rasa.utils.endpoints import EndpointConfig

import azure_search_store
from azure_search_store import AzureAISearch_Store, AzureSearchException

STUDENTS = "0006 Упутство за управљање подацима о ученицима.pdf"

ROLE_GROUPS = {
    "Administrator": ["*"],
    "Customer": ["public", "customer"],
    "Teacher": ["public", "teacher"],
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeAsyncClient:
    """Zamena za httpx.AsyncClient: pamti poslata tela i vraca zadate rezultate."""

    requests: List[Dict[str, Any]] = []
    hits: List[Dict[str, Any]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        FakeAsyncClient.requests.append(json)
        return FakeResponse({"value": list(FakeAsyncClient.hits)})


def tracker_state(roles) -> Dict[str, Any]:
    metadata = {} if roles is None else {"user_roles": roles}
    return {"latest_message": {"text": "pitanje", "metadata": metadata}}


def make_config(**overrides) -> EndpointConfig:
    kwargs = {
        "type": "azure_search_store.AzureAISearch_Store",
        "endpoint": "https://example.search.windows.net",
        "index_name": "indeks",
        "api_key": "kljuc",
        "content_field": "chunk",
        "metadata_fields": ["title"],
        "top_k": 4,
        "filter_field": "access_group",
        "role_groups": ROLE_GROUPS,
    }
    kwargs.update(overrides)
    return EndpointConfig(**{k: v for k, v in kwargs.items() if v is not None})


class AzureSearchRoleFilterTests(unittest.TestCase):
    def setUp(self):
        FakeAsyncClient.requests = []
        FakeAsyncClient.hits = [{"chunk": "ucenici 1", "title": STUDENTS}]
        patcher = mock.patch.object(azure_search_store.httpx, "AsyncClient", FakeAsyncClient)
        patcher.start()
        self.addCleanup(patcher.stop)

    def connected_store(self, **overrides) -> AzureAISearch_Store:
        store = AzureAISearch_Store(embeddings=None)
        store.connect(make_config(**overrides))
        return store

    def search(self, roles, store=None):
        store = store or self.connected_store()
        return asyncio.run(store.search("pitanje", tracker_state(roles)))

    def sent_filter(self):
        self.assertEqual(len(FakeAsyncClient.requests), 1)
        return FakeAsyncClient.requests[0].get("filter")

    # --- koji filter ide Azure-u ---

    def test_customer_gets_filter_with_its_groups(self):
        self.search(["Customer"])

        self.assertEqual(self.sent_filter(), "search.in(access_group, 'customer,public', ',')")

    def test_user_with_two_roles_gets_union_of_groups(self):
        self.search(["Customer", "Teacher"])

        self.assertEqual(self.sent_filter(), "search.in(access_group, 'customer,public,teacher', ',')")

    def test_administrator_searches_without_filter(self):
        self.search(["Administrator"])

        self.assertIsNone(self.sent_filter())

    def test_star_wins_when_combined_with_other_roles(self):
        self.search(["Customer", "Administrator"])

        self.assertIsNone(self.sent_filter())

    def test_unknown_role_adds_nothing_but_keeps_known_roles(self):
        self.search(["Customer", "Guest"])

        self.assertEqual(self.sent_filter(), "search.in(access_group, 'customer,public', ',')")

    def test_no_allowed_groups_means_no_azure_call_and_no_documents(self):
        store = self.connected_store()

        for roles in (None, [], ["Guest"], "Administrator", [7, ""]):
            with self.subTest(roles=roles):
                results = self.search(roles, store)
                self.assertEqual(results.results, [])
        self.assertEqual(FakeAsyncClient.requests, [])

    def test_vector_search_is_filtered_before_choosing_neighbours(self):
        store = self.connected_store(vector_field="text_vector")

        with mock.patch.object(azure_search_store, "aembed_query", mock.AsyncMock(return_value=[0.1, 0.2])):
            self.search(["Customer"], store)

        body = FakeAsyncClient.requests[0]
        self.assertEqual(body["filter"], "search.in(access_group, 'customer,public', ',')")
        self.assertEqual(body["vectorFilterMode"], "preFilter")
        self.assertEqual(body["vectorQueries"][0]["fields"], "text_vector")

    def test_administrator_vector_search_has_no_filter_mode(self):
        store = self.connected_store(vector_field="text_vector")

        with mock.patch.object(azure_search_store, "aembed_query", mock.AsyncMock(return_value=[0.1])):
            self.search(["Administrator"], store)

        body = FakeAsyncClient.requests[0]
        self.assertNotIn("filter", body)
        self.assertNotIn("vectorFilterMode", body)

    def test_results_keep_text_and_title(self):
        results = self.search(["Customer"])

        self.assertEqual([(r.text, r.metadata) for r in results.results], [("ucenici 1", {"title": STUDENTS})])

    # --- neispravna konfiguracija zaustavlja start ---

    def test_invalid_configuration_fails_connect(self):
        invalid = {
            "no filter_field": {"filter_field": None},
            "filter_field with OData": {"filter_field": "access_group or true"},
            "no role_groups": {"role_groups": None},
            "role_groups not a map": {"role_groups": ["Customer"]},
            "groups not a list": {"role_groups": {"Customer": "public"}},
            "empty groups": {"role_groups": {"Customer": []}},
            "uppercase group": {"role_groups": {"Customer": ["Public"]}},
            "quote in group": {"role_groups": {"Customer": ["public') or true or ('"]}},
            "comma in group": {"role_groups": {"Customer": ["public,admin"]}},
        }
        for name, overrides in invalid.items():
            with self.subTest(name):
                with self.assertRaises(AzureSearchException):
                    self.connected_store(**overrides)


if __name__ == "__main__":
    unittest.main()
