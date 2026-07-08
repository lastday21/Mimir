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
webview window. It also opens a compact always-on-top overlay for calls. The
overlay shows the latest detected question, streams the current answer, and can
pause or resume live audio capture.

Desktop hotkeys are `Ctrl+M` for showing or hiding the overlay and `Ctrl+Space`
for pausing or resuming live audio capture.

## Realtime Session Core

The main path is live speech detection followed by automatic answer streaming.
The backend exposes a session API:

- `POST /api/session/start`
- `POST /api/session/stop`
- `GET /api/session/events`
- `GET /api/audio/devices`
- `POST /api/session/audio/start`
- `POST /api/session/audio/stop`
- `POST /api/session/transcript`
- `POST /api/session/stt/wav`

`/api/session/audio/start` starts live capture for remote loopback and mic
audio. The default `yandex_realtime` mode streams remote loopback audio directly
to Yandex AI Studio Realtime API and uses mic SpeechKit transcripts only as
`MIC_CONTEXT`, so the assistant answers the other speaker and treats the user's
speech as dialogue context. `speechkit` mode remains available as a fallback
that sends both sources through the local SpeechKit transcript bus.

`/api/session/transcript` remains a development input for direct transcript
injection.

SpeechKit streaming uses direct SpeechKit v3 gRPC through the generated stubs
from the `yandexcloud` package. Realtime mode uses `aiohttp` WebSocket transport
with the `speech-realtime-250923` model.

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
npm run smoke
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
