import unittest

from mimir.question_detector import detect_questions


class QuestionDetectorTests(unittest.TestCase):
    def test_detects_russian_question_mark(self) -> None:
        questions = detect_questions("Как это работает?")
        self.assertEqual(len(questions), 1)
        self.assertGreaterEqual(questions[0].confidence, 0.9)

    def test_detects_russian_interrogative_without_question_mark(self) -> None:
        questions = detect_questions("А как вы решали такие задачи")
        self.assertEqual(len(questions), 1)
        self.assertGreaterEqual(questions[0].confidence, 0.6)

    def test_detects_russian_interview_prompt(self) -> None:
        questions = detect_questions("Расскажите о вашем опыте с Python")
        self.assertEqual(len(questions), 1)
        self.assertGreaterEqual(questions[0].confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
