import asyncio
import unittest

from mimir.providers.yandex_realtime import YandexRealtimeClient, YandexRealtimeConfig


class FakeWebSocket:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)


class YandexRealtimeClientTests(unittest.TestCase):
    def test_instruction_update_sends_only_session_patch(self) -> None:
        client = YandexRealtimeClient(YandexRealtimeConfig("test-key", "folder-id"))
        websocket = FakeWebSocket()
        client._ws = websocket

        asyncio.run(client.update_instructions("Новая цель разговора"))

        self.assertEqual(
            websocket.payloads,
            [
                {
                    "type": "session.update",
                    "session": {"instructions": "Новая цель разговора"},
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
