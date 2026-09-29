"""Periodicno brisanje starih podataka iz baze RasaTracker.

Pokrece ga ChatMessageBroker (chat_message_broker.py) u pozadinskoj niti, cim se
Rasa podigne i zatim na svakih `interval_minutes`. Brise:

1. `events` - interne dogadjaje ranijih sesija (pre poslednjeg
   `action_session_start`) starije od `events_retention_hours`. Rasa ih ionako
   ne ucitava, jer tracker cita samo poslednju sesiju.
2. `events` + `users` - ceo razgovor koji nema nijedan dogadjaj noviji od
   `events_retention_hours`. Ako se korisnik posle toga javi sa istim
   sender_id-jem, krece od pocetka (kao nova sesija, bez prenetih slotova).
3. `ChatMessages` - poruke starije od `messages_retention_days`
   (None ili 0 = poruke se nikad ne brisu).
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import sqlalchemy as sa
import structlog
from sqlalchemy.engine import Engine

logger = structlog.get_logger()

# 1. Dogadjaji iz ranijih sesija aktivnih razgovora.
# Ako razgovor nema action_session_start, podupit vraca NULL i nista se ne brise.
DELETE_OLD_SESSION_EVENTS = sa.text(
    """
    DELETE FROM events
    WHERE timestamp < :cutoff
      AND id < (
          SELECT MAX(s.id) FROM events s
          WHERE s.sender_id = events.sender_id
            AND s.type_name = 'action'
            AND s.action_name = 'action_session_start'
      )
    """
)

# 2. Razgovori bez aktivnosti posle granice: prvo mapiranje u `users`, pa dogadjaji
INACTIVE_SENDERS = """
    SELECT sender_id FROM events
    GROUP BY sender_id
    HAVING MAX(timestamp) < :cutoff
"""
DELETE_INACTIVE_USERS = sa.text(
    f"DELETE FROM users WHERE sender_id IN ({INACTIVE_SENDERS})"
)
DELETE_INACTIVE_EVENTS = sa.text(
    f"DELETE FROM events WHERE sender_id IN ({INACTIVE_SENDERS})"
)

# 3. Trajna istorija poruka (CreatedAt je UTC bez vremenske zone)
DELETE_OLD_MESSAGES = sa.text("DELETE FROM ChatMessages WHERE CreatedAt < :cutoff")


class HistoryCleanup:
    def __init__(
        self,
        engine: Engine,
        events_retention_hours: float = 24,
        messages_retention_days: Optional[float] = 90,
        interval_minutes: float = 60,
    ) -> None:
        self.engine = engine
        self.events_retention_hours = events_retention_hours
        self.messages_retention_days = messages_retention_days
        self.interval_minutes = interval_minutes
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        # daemon=True: nit ne sprecava gasenje Rase
        self._thread = threading.Thread(
            target=self._loop, name="history-cleanup", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        # Prvo ciscenje odmah pri startu, zatim na svakih interval_minutes
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval_minutes * 60)

    def run_once(self) -> None:
        # Greska pri ciscenju ne sme da obori Rasu - samo se loguje i pokusava opet
        try:
            # Tabele events/users pravi SQLTrackerStore; na potpuno novoj bazi
            # mozda jos ne postoje u trenutku prvog ciscenja
            if not sa.inspect(self.engine).has_table("events"):
                logger.debug("history_cleanup.skipped_no_events_table")
                return

            events_cutoff = time.time() - self.events_retention_hours * 3600
            with self.engine.begin() as conn:  # jedna transakcija, commit na kraju
                old_sessions = conn.execute(
                    DELETE_OLD_SESSION_EVENTS, {"cutoff": events_cutoff}
                ).rowcount
                conn.execute(DELETE_INACTIVE_USERS, {"cutoff": events_cutoff})
                inactive = conn.execute(
                    DELETE_INACTIVE_EVENTS, {"cutoff": events_cutoff}
                ).rowcount

                messages = 0
                if self.messages_retention_days:
                    messages_cutoff = datetime.now(timezone.utc).replace(
                        tzinfo=None
                    ) - timedelta(days=self.messages_retention_days)
                    messages = conn.execute(
                        DELETE_OLD_MESSAGES, {"cutoff": messages_cutoff}
                    ).rowcount

            logger.info(
                "history_cleanup.done",
                deleted_old_session_events=old_sessions,
                deleted_inactive_conversation_events=inactive,
                deleted_chat_messages=messages,
            )
        except Exception as e:
            logger.error("history_cleanup.failed", error=str(e))
