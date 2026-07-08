import json
import os
import tempfile
import unittest

from mimir.live_trace import current_trace_path, reset_trace_for_tests, trace_live_event


class LiveTraceTests(unittest.TestCase):
    def test_writes_jsonl_trace_and_redacts_secrets(self) -> None:
        original_dir = os.environ.get("MIMIR_LIVE_TRACE_DIR")
        original_enabled = os.environ.get("MIMIR_LIVE_TRACE")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["MIMIR_LIVE_TRACE_DIR"] = directory
                os.environ["MIMIR_LIVE_TRACE"] = "1"
                reset_trace_for_tests()

                trace_live_event("test.event", api_key="secret", text="hello", payload=b"123")

                path = current_trace_path()
                line = path.read_text(encoding="utf-8").strip()
                payload = json.loads(line)

                self.assertEqual(payload["event"], "test.event")
                self.assertEqual(payload["api_key"], "<redacted>")
                self.assertEqual(payload["text"], "hello")
                self.assertEqual(payload["payload"], {"bytes": 3})
        finally:
            if original_dir is None:
                os.environ.pop("MIMIR_LIVE_TRACE_DIR", None)
            else:
                os.environ["MIMIR_LIVE_TRACE_DIR"] = original_dir
            if original_enabled is None:
                os.environ.pop("MIMIR_LIVE_TRACE", None)
            else:
                os.environ["MIMIR_LIVE_TRACE"] = original_enabled
            reset_trace_for_tests()


if __name__ == "__main__":
    unittest.main()
