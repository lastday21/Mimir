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

Default desktop hotkeys are `Ctrl+M` for showing or hiding the overlay and
`Ctrl+Space` for pausing or resuming live audio capture. They can be changed in
the settings screen and take effect after restarting the desktop window. The
audio hotkey uses the mode selected in settings and shares the same audio control
path as the main window and overlay.

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
to Yandex AI Studio Realtime API and sends the recent role-labeled dialogue as
`DIALOGUE_CONTEXT`, so the assistant answers the other speaker and treats the
user's speech as dialogue history. `speechkit` mode remains available as a
fallback that sends both sources through the local SpeechKit transcript bus.
`local_vosk` mode is the offline path: local Vosk transcription feeds the same
session memory/question detector, while answers are forced through Ollama.
The settings screen exposes all three audio modes and keeps the local mode paired
with Ollama so the displayed configuration matches the runtime path.

If Yandex Realtime fails while starting or running, Mimir switches to
`local_vosk` with the same audio sources. This fallback is intended for quota or
token exhaustion cases where the call itself is still online but the Realtime
API cannot be used.

`/api/session/transcript` remains a development input for direct transcript
injection.

SpeechKit streaming uses direct SpeechKit v3 gRPC through the generated stubs
from the `yandexcloud` package. Realtime mode uses `aiohttp` WebSocket transport
with the `speech-realtime-250923` model.

Live audio writes a local JSONL trace to `.work/live-traces/`. The trace records
session state, transcripts, mic context, outgoing Realtime audio chunk sizes,
incoming Realtime events, answer deltas, and errors. It does not store API keys
or raw audio bytes. Set `MIMIR_LIVE_TRACE=0` to disable it or
`MIMIR_LIVE_TRACE_DIR=<path>` to write traces elsewhere.

The same trace includes local latency metrics as `metric.stage` and
`metric.question` events. Current in-memory metrics are also exposed at
`GET /api/metrics/current`. The tracked timings include audio chunk readiness,
STT interim/final text, question detection, context build, LLM first token,
first visible hint, and answer completion.

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

If Yandex AI Studio fails in the transcript-based LLM path, Mimir tries the
preferred local Ollama model before marking the session degraded. Realtime audio
mode still requires Yandex Realtime for the direct remote-audio path.

For the full offline audio fallback, install a local Ollama model and the Vosk
Russian streaming model:

```powershell
$env:OLLAMA_MODELS = "F:\MimirModels\ollama\models"
$env:MIMIR_VOSK_MODEL_PATH = "F:\MimirModels\vosk\vosk-model-small-ru-0.22"
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $env:OLLAMA_MODELS, "User")
[Environment]::SetEnvironmentVariable("MIMIR_VOSK_MODEL_PATH", $env:MIMIR_VOSK_MODEL_PATH, "User")
ollama pull qwen3:8b
.\.venv\Scripts\python.exe -m mimir.stt.local_vosk --install
```

The default local STT model is `vosk-model-small-ru-0.22`. Keep local models
outside the system drive when disk space is tight. On this workstation the
expected model root is:

```text
F:\MimirModels\
```

To use another Vosk model, set `MIMIR_VOSK_MODEL_PATH` to the unpacked model
directory before starting Mimir.
