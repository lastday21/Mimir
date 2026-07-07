# Mimir

Mimir is a Python-backed realtime assistant core for interviews and live calls.

The backend owns the session pipeline: dialogue memory, question/follow-up
triggering, context assembly, provider selection, and streaming answer events.
The current frontend is a development client for that pipeline.

## Desktop App

Use a Python 3.11+ interpreter for the virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
npm install
.\.venv\Scripts\python.exe -m mimir.desktop
```

The desktop launcher builds the React frontend when `dist/` is missing, starts
the Python API on a local random port, and opens the local client in a native
webview window.

## Realtime Session Core

The main path is no longer a manual record/detect/ask flow. The backend exposes
a session API:

- `POST /api/session/start`
- `POST /api/session/stop`
- `GET /api/session/events`
- `POST /api/session/transcript`
- `POST /api/session/stt/wav`
- `POST /api/manual/question`

`/api/session/transcript` is a development input until the continuous Python
audio capture path is wired in. It already feeds the same dialogue memory,
question trigger, context builder, and streaming LLM path that live audio will
use.

SpeechKit streaming uses direct SpeechKit v3 gRPC through the generated stubs
from the `yandexcloud` package.

`POST /api/session/stt/wav` accepts a mono 16-bit PCM WAV file as a development
feeder. It streams recognition results into the same session memory and question
trigger path that live mic/loopback capture will use.

## Development

Terminal 1:

```powershell
python -m mimir
```

Terminal 2:

```powershell
npm install
npm run dev
```

Open the URL shown by Vite.

## Built Frontend

```powershell
npm run build
python -m mimir
```

Open `http://127.0.0.1:8765`.

## Checks

```powershell
.\.venv\Scripts\python.exe -m scripts.check
npm run build
.\.venv\Scripts\python.exe -m mimir.desktop --check
```

## Credentials

Yandex keys are stored in Windows Credential Manager under:

- `Mimir:yandex_ai_studio`
- `Mimir:yandex_speechkit`

The local config file lives at:

```text
%APPDATA%\io.github.lastday21.mimir\config.json
```

## Local Fallback

For Ollama, Mimir prefers compact Qwen models first:

- `qwen3:8b`
- `qwen3:4b`
- Qwen 2.5 / Qwen 2 family models
