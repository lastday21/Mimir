import unittest

from mimir.providers.base import ProviderError
from mimir.providers.yandex_speechkit import YandexSpeechKitClient, normalize_language


class YandexSpeechKitTests(unittest.TestCase):
    def test_normalizes_short_language_codes(self) -> None:
        self.assertEqual(normalize_language("ru"), "ru-RU")
        self.assertEqual(normalize_language("en"), "en-US")
        self.assertEqual(normalize_language("tr"), "tr-TR")

    def test_rejects_missing_key_before_network_call(self) -> None:
        with self.assertRaises(ProviderError):
            YandexSpeechKitClient("").recognize_lpcm(b"\0\0")

    def test_rejects_unsupported_sample_rate(self) -> None:
        with self.assertRaisesRegex(ProviderError, "sample rate"):
            YandexSpeechKitClient("test").recognize_lpcm(
                b"\0\0",
                sample_rate_hertz=44_100,
            )


if __name__ == "__main__":
    unittest.main()
