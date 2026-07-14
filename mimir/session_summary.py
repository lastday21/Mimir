from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

from .models import ChatMessage
from .prompts import build_dialogue_summary_messages
from .session_types import AnswerStreamChunk


SUMMARY_UPDATE_TURNS = 6
SUMMARY_MAX_CHARS = 1600


class DialogueSummaryCoordinator:
    def __init__(
        self,
        stream_answer: Callable[[list[ChatMessage]], Iterator[AnswerStreamChunk | str]],
        load_config: Callable[[], Any],
        on_ready: Callable[[int, str, str, str, int, int], None],
        trace_event: Callable[..., None],
    ) -> None:
        self._stream_answer = stream_answer
        self._load_config = load_config
        self._on_ready = on_ready
        self._trace_event = trace_event
        self._lock = threading.Lock()
        self._generation = 0
        self._session_id = ""
        self._revision = 0
        self._requested_revision = 0
        self._thread: threading.Thread | None = None

    def reset(self, generation: int, session_id: str) -> None:
        with self._lock:
            self._generation = generation
            self._session_id = session_id
            self._revision = 0
            self._requested_revision = 0
            self._thread = None

    def observe(
        self,
        generation: int,
        session_id: str,
        previous_summary: str,
        transcript: str,
        through_turn_id: str,
        through_timestamp_ms: int,
    ) -> None:
        thread: threading.Thread | None = None
        with self._lock:
            if generation != self._generation or session_id != self._session_id:
                return
            self._revision += 1
            if self._thread is not None:
                return
            if self._revision - self._requested_revision < SUMMARY_UPDATE_TURNS:
                return
            if not transcript or not through_turn_id:
                return
            requested_revision = self._revision
            self._requested_revision = requested_revision
            thread = threading.Thread(
                target=self._run,
                args=(
                    generation,
                    session_id,
                    requested_revision,
                    previous_summary,
                    transcript,
                    through_turn_id,
                    through_timestamp_ms,
                ),
                name="mimir-dialogue-summary",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def _run(
        self,
        generation: int,
        session_id: str,
        requested_revision: int,
        previous_summary: str,
        transcript: str,
        through_turn_id: str,
        through_timestamp_ms: int,
    ) -> None:
        summary = ""
        error_text = ""
        try:
            messages = build_dialogue_summary_messages(
                previous_summary,
                transcript,
                self._load_config(),
            )
            parts: list[str] = []
            for chunk in self._stream_answer(messages):
                text = chunk.text if isinstance(chunk, AnswerStreamChunk) else str(chunk)
                if text:
                    parts.append(text)
            summary = "".join(parts).strip()[:SUMMARY_MAX_CHARS].rstrip()
        except Exception as error:
            error_text = str(error)

        with self._lock:
            is_current = generation == self._generation and session_id == self._session_id

        if is_current and summary:
            self._on_ready(
                generation,
                session_id,
                summary,
                through_turn_id,
                through_timestamp_ms,
                requested_revision,
            )

        current_thread = threading.current_thread()
        with self._lock:
            if self._thread is current_thread:
                self._thread = None
            if generation != self._generation or session_id != self._session_id:
                return
            if not summary:
                self._requested_revision = max(
                    0,
                    self._revision - SUMMARY_UPDATE_TURNS + 1,
                )

        if summary:
            self._trace_event(
                "session.context_summary",
                sessionId=session_id,
                revision=requested_revision,
                chars=len(summary),
            )
        else:
            self._trace_event(
                "session.context_summary_error",
                sessionId=session_id,
                error=error_text or "empty summary",
            )
