import { useEffect, useMemo, useState } from "react";
import { Bot, Check, Cloud, KeyRound, Loader2, MessageSquare, Play, Send, Server, Square, Zap } from "lucide-react";
import {
  AppConfig,
  ModelInfo,
  QuestionEvent,
  SessionSnapshot,
  TranscriptTurn,
  askManualQuestion,
  getConfig,
  listModels,
  saveConfig,
  sendTranscript,
  startLiveAudio,
  startSession,
  stopLiveAudio,
  stopSession,
  storeYandexKey,
  uploadSpeechWav
} from "./api";

const DEFAULT_CONFIG: AppConfig = {
  yandexFolderId: "",
  llmProvider: "yandex_ai_studio",
  llmModel: "yandexgpt/latest",
  ollamaBaseUrl: "http://localhost:11434",
  hasYandexKey: false
};

export function App() {
  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [apiKey, setApiKey] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [session, setSession] = useState<SessionSnapshot | null>(null);
  const [turns, setTurns] = useState<TranscriptTurn[]>([]);
  const [questions, setQuestions] = useState<QuestionEvent[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [source, setSource] = useState<"remote" | "mic">("remote");
  const [utterance, setUtterance] = useState("");
  const [manualQuestion, setManualQuestion] = useState("");
  const [wavFile, setWavFile] = useState<File | null>(null);
  const [liveRemote, setLiveRemote] = useState(true);
  const [liveMic, setLiveMic] = useState(true);
  const [audioRunning, setAudioRunning] = useState(false);
  const [status, setStatus] = useState("Start the Python API with python -m mimir");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getConfig()
      .then((loaded) => {
        setConfig(loaded);
        setStatus("Python API connected");
      })
      .catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    const events = new EventSource("/api/session/events");

    events.addEventListener("session_state", (event) => {
      const payload = parseEvent<SessionSnapshot>(event);
      setSession(payload);
      setStatus(`Session ${payload.state}`);
    });

    events.addEventListener("transcript", (event) => {
      const payload = parseEvent<TranscriptTurn>(event);
      setTurns((current) => [...current.slice(-39), payload]);
    });

    events.addEventListener("question", (event) => {
      const payload = parseEvent<QuestionEvent>(event);
      setQuestions((current) => [...current.slice(-9), payload]);
      setCurrentQuestion(payload.question);
      setAnswer("");
      setStatus(payload.reason === "manual" ? "Manual question sent" : "Question detected");
    });

    events.addEventListener("answer_delta", (event) => {
      const payload = parseEvent<{ deltaText: string }>(event);
      setAnswer((current) => current + payload.deltaText);
    });

    events.addEventListener("answer_done", () => {
      setStatus("Answer ready");
    });

    events.addEventListener("answer_cancelled", () => {
      setStatus("Previous answer cancelled");
    });

    events.addEventListener("answer_error", (event) => {
      const payload = parseEvent<{ error: string }>(event);
      setStatus(payload.error);
    });

    events.addEventListener("stt_status", (event) => {
      const payload = parseEvent<{ status: string; source: string }>(event);
      setStatus(`STT ${payload.source} ${payload.status}`);
    });

    events.addEventListener("stt_error", (event) => {
      const payload = parseEvent<{ error: string }>(event);
      setStatus(payload.error);
    });

    events.addEventListener("audio_status", (event) => {
      const payload = parseEvent<{ status: string; source?: string; running?: boolean }>(event);
      if (typeof payload.running === "boolean") {
        setAudioRunning(payload.running);
      }
      setStatus(payload.source ? `Audio ${payload.source} ${payload.status}` : `Audio ${payload.status}`);
    });

    events.addEventListener("audio_error", (event) => {
      const payload = parseEvent<{ error: string; running?: boolean }>(event);
      if (typeof payload.running === "boolean") {
        setAudioRunning(payload.running);
      }
      setStatus(payload.error);
    });

    events.onerror = () => {
      setStatus("Waiting for session events");
    };

    return () => events.close();
  }, []);

  const providerIcon = useMemo(() => {
    return config.llmProvider === "ollama" ? <Server size={18} /> : <Cloud size={18} />;
  }, [config.llmProvider]);

  async function persist(nextConfig = config) {
    const saved = await saveConfig(nextConfig);
    setConfig(saved);
    setStatus("Settings saved");
  }

  async function handleStoreKey() {
    if (!apiKey.trim()) return;
    setBusy(true);
    try {
      await storeYandexKey(apiKey.trim());
      setApiKey("");
      const loaded = await getConfig();
      setConfig(loaded);
      setStatus("Yandex key stored");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to store key");
    } finally {
      setBusy(false);
    }
  }

  async function handleLoadModels() {
    setBusy(true);
    try {
      await persist();
      const payload = await listModels();
      setModels(payload.models);
      if (payload.preferredModel) {
        const next = { ...config, llmModel: payload.preferredModel };
        setConfig(next);
        await persist(next);
      }
      setStatus(`Loaded ${payload.models.length} models`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to load models");
    } finally {
      setBusy(false);
    }
  }

  async function handleStartSession() {
    setBusy(true);
    try {
      const snapshot = await startSession();
      setSession(snapshot);
      setTurns(snapshot.memory.turns);
      setQuestions([]);
      setCurrentQuestion("");
      setAnswer("");
      setStatus("Session listening");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to start session");
    } finally {
      setBusy(false);
    }
  }

  async function handleStopSession() {
    setBusy(true);
    try {
      const snapshot = await stopSession();
      setSession(snapshot);
      setAudioRunning(false);
      setStatus("Session stopped");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to stop session");
    } finally {
      setBusy(false);
    }
  }

  async function handleStartLiveAudio() {
    const sources: Array<"remote" | "mic"> = [];
    if (liveRemote) sources.push("remote");
    if (liveMic) sources.push("mic");
    if (sources.length === 0) return;
    setBusy(true);
    try {
      const snapshot = await startLiveAudio(sources);
      setAudioRunning(snapshot.running);
      setStatus(`Audio streaming: ${snapshot.sources.join(" + ")}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to start audio");
    } finally {
      setBusy(false);
    }
  }

  async function handleStopLiveAudio() {
    setBusy(true);
    try {
      const snapshot = await stopLiveAudio();
      setAudioRunning(snapshot.running);
      setStatus("Audio stopped");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to stop audio");
    } finally {
      setBusy(false);
    }
  }

  async function handleSendTranscript() {
    if (!utterance.trim()) return;
    setBusy(true);
    try {
      await sendTranscript(source, utterance.trim(), true);
      setUtterance("");
      setStatus(source === "remote" ? "Remote turn added" : "Mic turn added");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to add turn");
    } finally {
      setBusy(false);
    }
  }

  async function handleManualQuestion() {
    if (!manualQuestion.trim()) return;
    setBusy(true);
    try {
      await askManualQuestion(manualQuestion.trim());
      setManualQuestion("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to ask question");
    } finally {
      setBusy(false);
    }
  }

  async function handleUploadWav() {
    if (!wavFile) return;
    setBusy(true);
    try {
      const payload = await uploadSpeechWav(source, wavFile);
      setWavFile(null);
      setStatus(`STT job started: ${payload.jobId}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to upload WAV");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header className="app-header">
        <div>
          <h1>Mimir</h1>
          <p>Python realtime session core for interview answers.</p>
        </div>
        <span className="status">{busy ? <Loader2 className="spin" size={16} /> : <Check size={16} />} {status}</span>
      </header>

      <section className="panel settings-panel">
        <div className="section-title">
          {providerIcon}
          <h2>Provider</h2>
        </div>
        <div className="settings-grid">
          <label>
            Provider
            <select
              value={config.llmProvider}
              onChange={(event) => setConfig({ ...config, llmProvider: event.target.value as AppConfig["llmProvider"] })}
            >
              <option value="yandex_ai_studio">Yandex AI Studio</option>
              <option value="ollama">Ollama</option>
            </select>
          </label>
          <label>
            Model
            <input value={config.llmModel} onChange={(event) => setConfig({ ...config, llmModel: event.target.value })} />
          </label>
          <label>
            Yandex folder ID
            <input value={config.yandexFolderId} onChange={(event) => setConfig({ ...config, yandexFolderId: event.target.value })} />
          </label>
          <label>
            Ollama URL
            <input value={config.ollamaBaseUrl} onChange={(event) => setConfig({ ...config, ollamaBaseUrl: event.target.value })} />
          </label>
        </div>

        <div className="key-row">
          <KeyRound size={18} />
          <input
            type="password"
            placeholder={config.hasYandexKey ? "Yandex key stored" : "Paste Yandex API key"}
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
          <button onClick={handleStoreKey} disabled={busy || !apiKey.trim()}>Store key</button>
          <button onClick={handleLoadModels} disabled={busy}>Load models</button>
          <button onClick={() => persist()} disabled={busy}>Save</button>
        </div>

        {models.length > 0 && (
          <select className="model-list" value={config.llmModel} onChange={(event) => setConfig({ ...config, llmModel: event.target.value })}>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name || model.id}{model.contextWindow ? ` (${Math.round(model.contextWindow / 1000)}K ctx)` : ""}
              </option>
            ))}
          </select>
        )}
      </section>

      <section className="panel session-panel">
        <div className="section-title">
          <Zap size={18} />
          <h2>Session</h2>
        </div>
        <div className="session-row">
          <button className="primary" onClick={handleStartSession} disabled={busy || session?.state === "listening" || session?.state === "answering"}>
            <Play size={16} />
            Start
          </button>
          <button className="danger" onClick={handleStopSession} disabled={busy || !session || session.state === "stopped"}>
            <Square size={16} />
            Stop
          </button>
          <span className="session-state">{session?.state ?? "idle"}</span>
          <span className="session-id">{session?.sessionId ?? "no session"}</span>
        </div>
        <div className="audio-row">
          <label className="check-row">
            <input type="checkbox" checked={liveRemote} onChange={(event) => setLiveRemote(event.target.checked)} />
            Remote loopback
          </label>
          <label className="check-row">
            <input type="checkbox" checked={liveMic} onChange={(event) => setLiveMic(event.target.checked)} />
            Mic
          </label>
          <button className="primary" onClick={handleStartLiveAudio} disabled={busy || audioRunning || (!liveRemote && !liveMic)}>
            <Play size={16} />
            Start audio
          </button>
          <button className="danger" onClick={handleStopLiveAudio} disabled={busy || !audioRunning}>
            <Square size={16} />
            Stop audio
          </button>
        </div>
      </section>

      <section className="workspace">
        <div className="panel">
          <div className="section-title">
            <MessageSquare size={18} />
            <h2>Transcript Bus</h2>
          </div>
          <div className="input-row">
            <select value={source} onChange={(event) => setSource(event.target.value as "remote" | "mic")}>
              <option value="remote">Remote</option>
              <option value="mic">Mic</option>
            </select>
            <button onClick={handleSendTranscript} disabled={busy || !utterance.trim()}>
              <Send size={16} />
              Add
            </button>
          </div>
          <textarea
            className="utterance-box"
            value={utterance}
            onChange={(event) => setUtterance(event.target.value)}
            placeholder="Type a remote or mic phrase for pipeline testing..."
          />
          <div className="wav-row">
            <input
              type="file"
              accept=".wav,audio/wav"
              onChange={(event) => setWavFile(event.target.files?.[0] ?? null)}
            />
            <button onClick={handleUploadWav} disabled={busy || !wavFile}>
              <Send size={16} />
              Stream WAV
            </button>
          </div>
          <div className="turns">
            {turns.map((turn, index) => (
              <div className={`turn ${turn.source}`} key={`${turn.timestampMs}-${index}`}>
                <strong>{turn.source === "remote" ? "Remote" : "Mic"}</strong>
                <span>{turn.text}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-title">
            <Bot size={18} />
            <h2>Answer Stream</h2>
          </div>
          <div className="manual-row">
            <input
              value={manualQuestion}
              onChange={(event) => setManualQuestion(event.target.value)}
              placeholder="Manual question override..."
            />
            <button className="primary" onClick={handleManualQuestion} disabled={busy || !manualQuestion.trim()}>
              <Send size={16} />
              Ask
            </button>
          </div>
          <article className="question-card">
            <small>Current question</small>
            <p>{currentQuestion || "No active question."}</p>
          </article>
          <article className="answer">{answer || "Streaming answer will appear here."}</article>
          <div className="questions">
            {questions.map((item) => (
              <button key={item.questionId} onClick={() => setManualQuestion(item.question)}>
                <span>{item.question}</span>
                <small>{Math.round(item.confidence * 100)}%</small>
              </button>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

function parseEvent<T>(event: Event): T {
  return JSON.parse((event as MessageEvent<string>).data) as T;
}
