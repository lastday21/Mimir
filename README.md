# Mimir

Mimir is a React and TypeScript meeting assistant with a Python backend.

The frontend runs through Vite. The backend exposes a small local HTTP API for
settings, model discovery, question detection, and assistant answers.

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
python -m scripts.check
npm run build
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
