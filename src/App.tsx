import { useEffect, useMemo, useState } from "react";
import { Bot, Check, Cloud, KeyRound, Loader2, MessageSquare, Server, Sparkles } from "lucide-react";
import {
  AppConfig,
  DetectedQuestion,
  ModelInfo,
  askMimir,
  detectQuestions,
  getConfig,
  listModels,
  saveConfig,
  storeYandexKey
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
  const [questions, setQuestions] = useState<DetectedQuestion[]>([]);
  const [transcript, setTranscript] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
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

  async function handleDetect() {
    setBusy(true);
    try {
      const payload = await detectQuestions(transcript);
      setQuestions(payload.questions);
      setStatus(`Detected ${payload.questions.length} questions`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Question detection failed");
    } finally {
      setBusy(false);
    }
  }

  async function handleAsk() {
    if (!question.trim()) return;
    setBusy(true);
    setAnswer("");
    try {
      await persist();
      const payload = await askMimir(question, transcript);
      setAnswer(payload.answer);
      setStatus("Answer ready");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header className="app-header">
        <div>
          <h1>Mimir</h1>
          <p>React/TypeScript frontend with a Python assistant backend.</p>
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

      <section className="workspace">
        <div className="panel">
          <div className="section-title">
            <MessageSquare size={18} />
            <h2>Transcript</h2>
          </div>
          <textarea
            value={transcript}
            onChange={(event) => setTranscript(event.target.value)}
            placeholder="Paste meeting or interview transcript here..."
          />
          <button onClick={handleDetect} disabled={busy}>Detect questions</button>
          <div className="questions">
            {questions.map((item, index) => (
              <button key={`${item.text}-${index}`} onClick={() => setQuestion(item.text)}>
                <span>{item.text}</span>
                <small>{Math.round(item.confidence * 100)}%</small>
              </button>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-title">
            <Bot size={18} />
            <h2>Ask Mimir</h2>
          </div>
          <textarea
            className="question-box"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question..."
          />
          <button className="primary" onClick={handleAsk} disabled={busy || !question.trim()}>
            <Sparkles size={16} />
            Send
          </button>
          <article className="answer">{answer || "The answer will appear here."}</article>
        </div>
      </section>
    </main>
  );
}
