import threading
import time
import unittest

from mimir.config import AppConfig
from mimir.session_summary import DialogueSummaryCoordinator


class DialogueSummaryCoordinatorTests(unittest.TestCase):
    def test_new_turn_during_summary_triggers_immediate_catch_up(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        ready: list[tuple[str, int]] = []
        calls = 0

        def stream_answer(_messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                release_first.wait(timeout=2)
                yield "Первая сводка"
                return
            yield "Свежая сводка"

        coordinator = DialogueSummaryCoordinator(
            stream_answer,
            AppConfig,
            lambda _generation, _session_id, summary, turn_id, _timestamp, _revision: ready.append(
                (f"{summary}:{turn_id}", _revision)
            ),
            lambda *_args, **_kwargs: None,
        )
        coordinator.reset(1, "session_test")

        for revision in range(1, 7):
            coordinator.observe(
                1,
                "session_test",
                "",
                f"Реплики по {revision}",
                f"turn_{revision}",
                revision,
            )

        self.assertTrue(first_started.wait(timeout=1))
        coordinator.observe(
            1,
            "session_test",
            "",
            "Реплики по 7",
            "turn_7",
            7,
        )
        release_first.set()

        deadline = time.monotonic() + 2
        while len(ready) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(calls, 2)
        self.assertEqual(ready, [("Свежая сводка:turn_7", 7)])

    def test_forced_correction_updates_already_completed_summary(self) -> None:
        ready: list[tuple[str, int]] = []
        calls = 0

        def stream_answer(_messages):
            nonlocal calls
            calls += 1
            yield f"Сводка {calls}"

        coordinator = DialogueSummaryCoordinator(
            stream_answer,
            AppConfig,
            lambda _generation, _session_id, summary, _turn_id, _timestamp, revision: ready.append(
                (summary, revision)
            ),
            lambda *_args, **_kwargs: None,
        )
        coordinator.reset(1, "session_test")

        for revision in range(1, 7):
            coordinator.observe(
                1,
                "session_test",
                "",
                f"Реплики по {revision}",
                f"turn_{revision}",
                revision,
            )

        deadline = time.monotonic() + 2
        while len(ready) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        coordinator.observe(
            1,
            "session_test",
            ready[-1][0],
            "Исправленные реплики",
            "turn_5",
            5,
            force=True,
        )

        deadline = time.monotonic() + 2
        while len(ready) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(calls, 2)
        self.assertEqual(ready[-1], ("Сводка 2", 7))

    def test_early_refinement_does_not_force_premature_summary(self) -> None:
        calls = 0

        def stream_answer(_messages):
            nonlocal calls
            calls += 1
            yield "Сводка"

        coordinator = DialogueSummaryCoordinator(
            stream_answer,
            AppConfig,
            lambda *_args: True,
            lambda *_args, **_kwargs: None,
        )
        coordinator.reset(1, "session_test")
        coordinator.observe(
            1,
            "session_test",
            "",
            "Первая уточненная реплика",
            "turn_1",
            1,
            force=True,
        )

        time.sleep(0.05)

        self.assertEqual(calls, 0)

    def test_rejected_summary_is_not_chained_into_follow_up(self) -> None:
        rejection_started = threading.Event()
        release_rejection = threading.Event()
        prompts: list[str] = []
        ready: list[str] = []

        def stream_answer(messages):
            prompt = messages[-1].content
            prompts.append(prompt)
            if len(prompts) == 1:
                yield "Опасная сводка"
                return
            yield "Безопасная сводка"

        def on_ready(_generation, _session_id, summary, _turn_id, _timestamp, _revision):
            if summary == "Опасная сводка":
                rejection_started.set()
                release_rejection.wait(timeout=2)
                return False
            ready.append(summary)
            return True

        coordinator = DialogueSummaryCoordinator(
            stream_answer,
            AppConfig,
            on_ready,
            lambda *_args, **_kwargs: None,
        )
        coordinator.reset(1, "session_test")
        for revision in range(1, 7):
            coordinator.observe(
                1,
                "session_test",
                "Старая безопасная сводка",
                f"Реплики по {revision}",
                f"turn_{revision}",
                revision,
            )

        self.assertTrue(rejection_started.wait(timeout=1))
        coordinator.observe(
            1,
            "session_test",
            "Старая безопасная сводка",
            "Новая нормальная реплика",
            "turn_7",
            7,
        )
        release_rejection.set()

        deadline = time.monotonic() + 2
        while not ready and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(ready, ["Безопасная сводка"])
        self.assertNotIn("Опасная сводка", prompts[-1])
        self.assertIn("Старая безопасная сводка", prompts[-1])


if __name__ == "__main__":
    unittest.main()
