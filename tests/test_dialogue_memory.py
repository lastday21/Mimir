import unittest

from mimir.dialogue import MIC_SOURCE, REMOTE_SOURCE, DialogueMemory, DialogueTurn


class DialogueMemoryTests(unittest.TestCase):
    def test_context_keeps_remote_and_user_history_for_follow_up(self) -> None:
        memory = DialogueMemory()
        memory.append(DialogueTurn(REMOTE_SOURCE, "Расскажите, как вы проектировали очередь задач"))
        memory.append(DialogueTurn(MIC_SOURCE, "Я разделял задачи по приоритетам и обрабатывал их воркерами"))
        memory.remember_question("Расскажите, как вы проектировали очередь задач")

        context = memory.build_context(
            session_id="session_test",
            question_id="question_test",
            question="А если нагрузка вырастет в 10 раз?",
            confidence=0.9,
        )

        self.assertIn("очередь", context.transcript_excerpt)
        self.assertIn("воркерами", context.transcript_excerpt)
        self.assertEqual(context.relevant_prior_questions[-1], "Расскажите, как вы проектировали очередь задач")

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
