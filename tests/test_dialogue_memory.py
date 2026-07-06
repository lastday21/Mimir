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


if __name__ == "__main__":
    unittest.main()
