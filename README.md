# Mimir

Mimir is a React and TypeScript meeting assistant with a Python backend.

The frontend runs through Vite. The backend exposes a small local HTTP API for
settings, model discovery, question detection, and assistant answers.

## Desktop App

Use a Python 3.11+ interpreter for the virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
npm install
.\.venv\Scripts\python.exe -m mimir.desktop
```

The desktop launcher builds the React frontend when `dist/` is missing, starts
the Python API on a local random port, and opens Mimir in a native webview
window.

## Speech Recognition

The transcript panel can record up to 30 seconds from the microphone and send
16 kHz mono LPCM audio to Yandex SpeechKit. The recognized text is appended to
the transcript and can then be used for question detection or Ask Mimir.

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
