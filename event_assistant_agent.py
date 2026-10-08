"""ReAct sub agent `event_assistant` (konfiguracija: sub_agents/event_assistant/config.yml).

Ugradjeni MCPOpenAgent (LLM petlja: razmisli -> pozovi alat -> razmisli -> odgovori),
prosiren samo za prenos identiteta korisnika do MCP alata (mcp_tools/server.py):

1. process_input - javni Rasa hook, poziva se pre svakog pokretanja agenta.
   Iz poslednje korisnikove poruke uzima `user_token` (upisuje ga secure_rest
   kanal tek posto ga proveri) i stavlja ga u agent_input.metadata. Metadata
   se ne prikazuje LLM-u (nije u promptu), za razliku od slotova.
2. _get_meta_for_mcp_server - INTERNI Rasa metod (rasa-pro 3.19). Na `_meta`
   svakog MCP poziva dodaje `user_token`; LLM ga ne vidi i ne moze da ga menja.
   Ako ga Rasa u novoj verziji promeni, token se vise ne salje, .NET odbija
   pozive (401) i agent prestaje da radi - ali ne otkriva tudje podatke.
   Zato je rasa-pro zakucan na 3.19.* (requirements.txt), a test
   tests/test_event_assistant_agent.py pada ako se metod promeni.
"""

from typing import Any, Dict, Optional

from rasa.agents.protocol.mcp.mcp_open_agent import MCPOpenAgent
from rasa.agents.schemas import AgentInput
from rasa.agents.schemas.agent_input import AgentInputSlot
from rasa.shared.core.events import SlotSet, UserUttered

USER_TOKEN_KEY = "user_token"
LANGUAGE_SLOT = "language"


def latest_user_token(agent_input: AgentInput) -> Optional[str]:
    """Token iz metadata poslednje korisnikove poruke (svaka poruka nosi nov token)."""
    for event in reversed(agent_input.events):
        if isinstance(event, UserUttered):
            return (event.metadata or {}).get(USER_TOKEN_KEY) or None
    return None


def latest_language(agent_input: AgentInput) -> Optional[str]:
    """Jezik razgovora iz poslednjeg SlotSet(language) dogadjaja."""
    for event in reversed(agent_input.events):
        if isinstance(event, SlotSet) and event.key == LANGUAGE_SLOT and event.value:
            return str(event.value)
    return None


class EventAssistantAgent(MCPOpenAgent):
    async def process_input(self, input: AgentInput) -> AgentInput:
        # Uvek se postavlja iznova: stari token ne sme da ostane ako nova poruka nema token.
        input.metadata[USER_TOKEN_KEY] = latest_user_token(input)

        # Slot `language` se ne prosledjuje agentu ako ga nema u domenu; prompt ga
        # koristi za jezik odgovora, pa se ovde dodaje iz dogadjaja ako nedostaje.
        if LANGUAGE_SLOT not in input.slot_names:
            language = latest_language(input)
            if language:
                input.slots.append(
                    AgentInputSlot(name=LANGUAGE_SLOT, value=language, type="text")
                )
        return input

    def _get_meta_for_mcp_server(
        self, server_id: str, agent_input: Optional[AgentInput]
    ) -> Dict[str, Any]:
        meta = dict(super()._get_meta_for_mcp_server(server_id, agent_input))
        token = (agent_input.metadata or {}).get(USER_TOKEN_KEY) if agent_input else None
        if token:
            meta[USER_TOKEN_KEY] = token
        return meta
