"""Testovi za event_assistant_agent.py i prompt agenta.

Najvazniji je test_token_reaches_mcp_meta_through_rasas_tool_call: prolazi kroz
Rasin pravi kod za poziv MCP alata (_execute_mcp_tool) i proverava da token
stigne u _meta. Ako nova verzija Rase promeni interni _get_meta_for_mcp_server,
ovaj test pada (umesto da agent tiho prestane da radi).

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import inspect
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from jinja2 import Template
from mcp.types import CallToolResult, TextContent

from rasa.agents.protocol.mcp import mcp_base_agent
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import AgentInput
from rasa.agents.schemas.agent_input import AgentInputSlot
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered

from event_assistant_agent import EventAssistantAgent

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "event_assistant_prompt.jinja2"


def make_input(events: List[Any], slots: List[AgentInputSlot] = None, metadata: Dict = None) -> AgentInput:
    return AgentInput(
        id="event_assistant",
        user_message="",
        slots=slots or [],
        conversation_history="",
        events=events,
        metadata=metadata if metadata is not None else {},
    )


def make_agent() -> EventAssistantAgent:
    # Bez konstruktora: on pravi pravi LLM klijent. Postavljaju se samo polja
    # koja koristi putanja za poziv MCP alata.
    agent = object.__new__(EventAssistantAgent)
    agent._name = "event_assistant"
    agent._server_to_meta_map = {}
    agent._server_to_pre_call_hook = {}
    agent._tool_timeout = 20
    return agent


class ProcessInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_takes_token_from_latest_user_message(self) -> None:
        agent_input = make_input([
            UserUttered("prva", metadata={"user_token": "old"}),
            BotUttered("odgovor"),
            UserUttered("druga", metadata={"user_token": "new"}),
            BotUttered("trenutak"),
        ])

        result = await make_agent().process_input(agent_input)

        self.assertEqual(result.metadata["user_token"], "new")

    async def test_stale_token_is_cleared_when_latest_message_has_none(self) -> None:
        agent_input = make_input(
            [UserUttered("poruka", metadata={})], metadata={"user_token": "stale"}
        )

        result = await make_agent().process_input(agent_input)

        self.assertIsNone(result.metadata["user_token"])

    async def test_adds_language_slot_from_events_when_missing(self) -> None:
        agent_input = make_input([SlotSet("language", "sr-Cyrl"), SlotSet("language", "en"), UserUttered("hi")])

        result = await make_agent().process_input(agent_input)

        self.assertEqual([(s.name, s.value) for s in result.slots], [("language", "en")])

    async def test_keeps_existing_language_slot(self) -> None:
        agent_input = make_input(
            [SlotSet("language", "en"), UserUttered("zdravo")],
            slots=[AgentInputSlot(name="language", value="sr-Cyrl", type="text")],
        )

        result = await make_agent().process_input(agent_input)

        self.assertEqual([(s.name, s.value) for s in result.slots], [("language", "sr-Cyrl")])


class McpMetaTests(unittest.IsolatedAsyncioTestCase):
    def test_rasa_internal_method_still_has_expected_signature(self) -> None:
        # Interni Rasa metod koji prepisujemo - ako se promeni, prilagoditi event_assistant_agent.py.
        params = list(inspect.signature(MCPBaseAgent._get_meta_for_mcp_server).parameters)
        self.assertEqual(params, ["self", "server_id", "agent_input"])

    def test_meta_contains_token_from_metadata(self) -> None:
        agent_input = make_input([], metadata={"user_token": "tok"})

        meta = make_agent()._get_meta_for_mcp_server("event_tools", agent_input)

        self.assertEqual(meta, {"user_token": "tok"})

    def test_no_token_means_no_meta(self) -> None:
        self.assertEqual(make_agent()._get_meta_for_mcp_server("event_tools", make_input([])), {})
        self.assertEqual(make_agent()._get_meta_for_mcp_server("event_tools", None), {})

    async def test_token_reaches_mcp_meta_through_rasas_tool_call(self) -> None:
        agent = make_agent()
        connection = mock.Mock(server_url="http://localhost:8765/mcp")
        connection.ensure_active_session = mock.AsyncMock(return_value="session")
        agent._tool_to_server_mapper = {"get_my_reservations": "event_tools"}
        agent._server_connections = {"event_tools": connection}

        agent_input = await agent.process_input(
            make_input([UserUttered("moje rezervacije", metadata={"user_token": "tok-xyz"})])
        )
        sent = mock.AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="[]")]))
        with mock.patch.object(mcp_base_agent, "call_tool_with_meta", sent):
            result = await agent._execute_mcp_tool("get_my_reservations", {}, agent_input)

        self.assertFalse(result.is_error, result.error_message)
        meta = sent.call_args.args[4]  # call_tool_with_meta(session, tool, args, timeout, meta)
        self.assertEqual(meta["user_token"], "tok-xyz")


class PromptTests(unittest.TestCase):
    def render(self, **kwargs: Any) -> str:
        values = {"description": "TASK", "slots": {}, "current_datetime": None,
                  "resumed_after_interruption": False, "restarted": False}
        values.update(kwargs)
        return Template(PROMPT_PATH.read_text(encoding="utf-8")).render(**values)

    def test_language_from_slot(self) -> None:
        self.assertIn("ENTIRE reply in English", self.render(slots={"language": "en"}))

    def test_serbian_latin(self) -> None:
        self.assertIn(
            "ENTIRE reply in Serbian, written in Latin script", self.render(slots={"language": "sr-Latn"})
        )

    def test_default_language_is_serbian_cyrillic(self) -> None:
        self.assertIn("ENTIRE reply in Serbian, written in Cyrillic script", self.render())

    def test_language_is_repeated_at_the_end(self) -> None:
        last_line = self.render(slots={"language": "en"}).strip().splitlines()[-1]
        self.assertIn("Reminder: write your whole reply in English", last_line)

    def test_formal_serbian_only_for_serbian(self) -> None:
        self.assertIn('never "ti"', self.render(slots={"language": "sr-Latn"}))
        self.assertNotIn('never "ti"', self.render(slots={"language": "en"}))

    def test_description_and_key_rules_present(self) -> None:
        prompt = self.render()
        self.assertIn("TASK", prompt)
        self.assertIn("EXACTLY as the tools returned them", prompt)
        self.assertIn("check EVERY pair", prompt)
        self.assertIn("build it straight away", prompt)
        self.assertIn("do NOT call `task_completed`", prompt)


if __name__ == "__main__":
    unittest.main()
