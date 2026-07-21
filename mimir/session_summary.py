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
        on_ready: Callable[[int, str, str, str, int, int], bool | None],
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
        self._latest_observation: tuple[str, str, str, int] | None = None

    def reset(self, generation: int, session_id: str) -> None:
        with self._lock:
            self._generation = generation
            self._session_id = session_id
            self._revision = 0
            self._requested_revision = 0
            self._thread = None
            self._latest_observation = None

    def observe(
        self,
        generation: int,
        session_id: str,
        previous_summary: str,
        transcript: str,
        through_turn_id: str,
        through_timestamp_ms: int,
        *,
        force: bool = False,
    ) -> None:
        thread: threading.Thread | None = None
        with self._lock:
            if generation != self._generation or session_id != self._session_id:
                return
            self._revision += 1
            self._latest_observation = (
                previous_summary,
                transcript,
                through_turn_id,
                through_timestamp_ms,
            )
            if self._thread is not None:
                return
            should_force = force and (
                bool(previous_summary.strip())
                or self._requested_revision > 0
            )
            if not should_force and self._revision - self._requested_revision < SUMMARY_UPDATE_TURNS:
                return
            if not transcript or not through_turn_id:
                return
            requested_revision = self._revision
            self._requested_revision = requested_revision
            thread = self._build_thread_locked(
                generation,
                session_id,
                requested_revision,
                previous_summary,
                transcript,
                through_turn_id,
                through_timestamp_ms,
            )
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

        summary_accepted = False
        with self._lock:
            is_current = generation == self._generation and session_id == self._session_id
            is_latest = is_current and self._revision == requested_revision

        if is_latest and summary:
            accepted = self._on_ready(
                generation,
                session_id,
                summary,
                through_turn_id,
                through_timestamp_ms,
                requested_revision,
            )
            summary_accepted = accepted is not False

        current_thread = threading.current_thread()
        follow_up: threading.Thread | None = None
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
            elif self._revision > requested_revision and self._latest_observation is not None:
                (
                    latest_previous_summary,
                    latest_transcript,
                    latest_turn_id,
                    latest_timestamp_ms,
                ) = self._latest_observation
                latest_revision = self._revision
                self._requested_revision = latest_revision
                follow_up_previous_summary = (
                    latest_previous_summary
                    if (
                        not summary_accepted
                        or latest_timestamp_ms < through_timestamp_ms
                    )
                    else summary
                )
                follow_up = self._build_thread_locked(
                    generation,
                    session_id,
                    latest_revision,
                    follow_up_previous_summary,
                    latest_transcript,
                    latest_turn_id,
                    latest_timestamp_ms,
                )

        if summary and is_latest:
            self._trace_event(
                "session.context_summary",
                sessionId=session_id,
                revision=requested_revision,
                chars=len(summary),
            )
        elif summary:
            self._trace_event(
                "session.context_summary_stale",
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
        if follow_up is not None:
            follow_up.start()

    def _build_thread_locked(
        self,
        generation: int,
        session_id: str,
        requested_revision: int,
        previous_summary: str,
        transcript: str,
        through_turn_id: str,
        through_timestamp_ms: int,
    ) -> threading.Thread:
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
        return thread
