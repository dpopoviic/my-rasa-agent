"""Event broker koji trajno cuva SAMO poruke (korisnik / bot) u tabelu ChatMessages.

Rasa svaki dogadjaj iz tracker store-a salje i event broker-u. Ovaj broker
propusta samo `user` i `bot` dogadjaje, pa po poruci nastaje jedan red (~0,5 KB)
umesto ~47 internih redova u tabeli `events`. Tabelu `events` Rasa i dalje koristi
kao radnu memoriju i ona se periodicno brise; ChatMessages je trajna istorija.
"""

from asyncio import AbstractEventLoop
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Text

import sqlalchemy as sa
import structlog
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from rasa.core.brokers.broker import EventBroker
from rasa.core.tracker_stores.sql_tracker_store import SQLTrackerStore
from rasa.utils.endpoints import EndpointConfig

logger = structlog.get_logger()

# Rasa tipovi dogadjaja koji se cuvaju -> vrednost kolone Role
SAVED_EVENTS = {"user": "user", "bot": "bot"}


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "ChatMessages"

    Id = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    SenderId = sa.Column(sa.Unicode(255), nullable=False)  # ID razgovora (sender_id)
    UserId = sa.Column(sa.Unicode(255), nullable=True, index=True)  # metadata.user_id
    Role = sa.Column(sa.Unicode(10), nullable=False)  # 'user' ili 'bot'
    # Unicode (NVARCHAR) da se cirilica ispravno sacuva
    Text = sa.Column(sa.UnicodeText, nullable=False)
    CreatedAt = sa.Column(sa.DateTime, nullable=False, index=True)  # UTC

    # Brzo citanje jednog razgovora po redosledu poruka
    __table_args__ = (sa.Index("IX_ChatMessages_SenderId_Id", "SenderId", "Id"),)


class ChatMessageBroker(EventBroker):
    def __init__(
        self,
        dialect: Text = "mssql+pyodbc",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "RasaTracker",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        query: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        # Isti nacin gradjenja konekcije kao SQLTrackerStore (podrzava query/driver za MSSQL)
        engine_url = SQLTrackerStore.get_db_url(
            dialect, host, port, db, username, password, query=query
        )
        self.engine = sa.create_engine(engine_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)  # pravi tabelu ChatMessages ako ne postoji
        self.sessionmaker = sessionmaker(bind=self.engine)
        logger.debug("chat_message_broker.connected", db=db)

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> "ChatMessageBroker":
        return cls(host=broker_config.url, **broker_config.kwargs)

    def publish(self, event: Dict[Text, Any]) -> None:
        role = SAVED_EVENTS.get(event.get("event"))
        text = event.get("text")
        if not role or not text:
            return  # interni dogadjaj (flow, slot, akcija...) ili poruka bez teksta

        timestamp = event.get("timestamp") or datetime.now(timezone.utc).timestamp()
        message = ChatMessage(
            SenderId=event.get("sender_id"),
            UserId=event.get("user_id"),
            Role=role,
            Text=text,
            CreatedAt=datetime.fromtimestamp(timestamp, timezone.utc).replace(
                tzinfo=None
            ),
        )
        # Greska pri upisu istorije ne sme da prekine razgovor - samo se loguje
        try:
            with self.sessionmaker() as session:
                session.add(message)
                session.commit()
        except Exception as e:
            logger.error(
                "chat_message_broker.publish_failed",
                sender_id=event.get("sender_id"),
                error=str(e),
            )

    async def close(self) -> None:
        self.engine.dispose()
