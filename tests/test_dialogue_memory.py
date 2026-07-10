import unittest

from mimir.dialogue import (
    MIC_SOURCE,
    MIN_MEMORY_WINDOW_MS,
    REMOTE_SOURCE,
    DialogueMemory,
    DialogueTurn,
)


class DialogueMemoryTests(unittest.TestCase):
    def test_context_keeps_linked_question_hint_and_user_answer(self) -> None:
        now = [1_000_000]
        memory = DialogueMemory(clock_ms=lambda: now[0])
        memory.append(DialogueTurn(REMOTE_SOURCE, "Расскажите, как вы проектировали очередь задач", timestamp_ms=now[0]))
        memory.remember_question("question_queue", "Расскажите, как вы проектировали очередь задач", now[0])
        memory.record_hint_delta("question_queue", "Сначала опишите разделение по приоритетам.", now[0] + 1)
        answer = memory.append(
            DialogueTurn(
                MIC_SOURCE,
                "Я разделял задачи по приоритетам и обрабатывал их воркерами",
                timestamp_ms=now[0] + 2,
            )
        )
        self.assertIsNotNone(answer)
        memory.record_user_answer("question_queue", answer.turn)

        context = memory.build_context(
            session_id="session_test",
            question_id="question_scale",
            question="А если нагрузка вырастет в 10 раз?",
            confidence=0.9,
        )
        prompt = context.to_prompt_text()

        self.assertIn("очередь", context.transcript_excerpt)
        self.assertIn("воркерами", context.transcript_excerpt)
        self.assertEqual(context.relevant_prior_questions[-1], "Расскажите, как вы проектировали очередь задач")
        self.assertIn("Подсказка Mimir: Сначала опишите разделение", prompt)
        self.assertIn("Ответ пользователя: Я разделял задачи", prompt)

    def test_interim_turn_is_replaced_by_next_interim_and_final(self) -> None:
        now = [1_000_000]
        memory = DialogueMemory(clock_ms=lambda: now[0])

        first = memory.append(DialogueTurn(REMOTE_SOURCE, "Как", is_final=False, timestamp_ms=now[0]))
        second = memory.append(DialogueTurn(REMOTE_SOURCE, "Как вы", is_final=False, timestamp_ms=now[0] + 1))
        final = memory.append(DialogueTurn(REMOTE_SOURCE, "Как вы строили сервис?", timestamp_ms=now[0] + 2))

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(final)
        self.assertEqual(first.operation, "append")
        self.assertEqual(second.operation, "replace")
        self.assertEqual(final.operation, "replace")
        self.assertEqual({first.turn.turn_id, second.turn.turn_id, final.turn.turn_id}, {first.turn.turn_id})
        self.assertEqual(len(memory.turns), 1)
        self.assertTrue(memory.turns[0].is_final)
        self.assertEqual(memory.turns[0].text, "Как вы строили сервис?")

    def test_final_refinement_replaces_final_turn_and_linked_user_answer(self) -> None:
        now = [1_000_000]
        memory = DialogueMemory(clock_ms=lambda: now[0])
        memory.remember_question("question_one", "Расскажите про проект", now[0])
        first = memory.append(DialogueTurn(MIC_SOURCE, "делал сервис", timestamp_ms=now[0] + 1))
        self.assertIsNotNone(first)
        memory.record_user_answer("question_one", first.turn)

        refined = memory.append(
            DialogueTurn(MIC_SOURCE, "Я делал сервис", timestamp_ms=now[0] + 2),
            refine_latest=True,
        )
        self.assertIsNotNone(refined)
        memory.record_user_answer("question_one", refined.turn)
        exchange = memory.payload()["exchanges"][0]

        self.assertEqual(refined.operation, "replace")
        self.assertEqual(refined.turn.turn_id, first.turn.turn_id)
        self.assertEqual(len(memory.turns), 1)
        self.assertEqual(exchange["userAnswer"], "Я делал сервис")

    def test_memory_uses_at_least_five_minute_window_instead_of_turn_count(self) -> None:
        now = [1_000_000]
        memory = DialogueMemory(retention_ms=1_000, clock_ms=lambda: now[0])
        self.assertEqual(memory.retention_ms, MIN_MEMORY_WINDOW_MS)

        memory.append(
            DialogueTurn(REMOTE_SOURCE, "Слишком старая реплика", timestamp_ms=now[0] - MIN_MEMORY_WINDOW_MS - 1)
        )
        memory.append(
            DialogueTurn(REMOTE_SOURCE, "Граница окна", timestamp_ms=now[0] - MIN_MEMORY_WINDOW_MS)
        )
        for index in range(100):
            memory.append(DialogueTurn(MIC_SOURCE, f"Реплика {index}", timestamp_ms=now[0] + index))

        payload = memory.payload()

        self.assertEqual(payload["windowMs"], MIN_MEMORY_WINDOW_MS)
        self.assertNotIn("Слишком старая реплика", [turn["text"] for turn in payload["turns"]])
        self.assertIn("Граница окна", [turn["text"] for turn in payload["turns"]])
        self.assertEqual(len(payload["turns"]), 101)

    def test_realtime_context_keeps_role_labeled_final_turns(self) -> None:
        memory = DialogueMemory()
        memory.append(DialogueTurn(REMOTE_SOURCE, "интерим", is_final=False))
        memory.append(DialogueTurn(REMOTE_SOURCE, "Расскажите про Kafka"))
        memory.append(DialogueTurn(MIC_SOURCE, "угу"))
        memory.append(DialogueTurn(MIC_SOURCE, "Я использовал Kafka для событий между сервисами"))

        context = memory.realtime_context(max_turns=3, max_chars=500)

        self.assertNotIn("интерим", context)
        self.assertIn("Собеседник: Расскажите про Kafka", context)
        self.assertIn("Пользователь: угу", context)
        self.assertIn("Пользователь: Я использовал Kafka", context)


if __name__ == "__main__":
    unittest.main()
