import { useEffect, useMemo, useRef, useState } from "react";
import { Bot, Check, Cloud, KeyRound, Loader2, MessageSquare, Mic, Server, Sparkles, Square } from "lucide-react";
import {
  AppConfig,
  DetectedQuestion,
  ModelInfo,
  askMimir,
  detectQuestions,
  getConfig,
  listModels,
  saveConfig,
  storeYandexKey,
  transcribeYandexSpeech
} from "./api";

const STT_SAMPLE_RATE = 16000;
const MAX_RECORDING_SECONDS = 30;

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
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [sttLanguage, setSttLanguage] = useState("ru-RU");

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const inputSampleRateRef = useRef(STT_SAMPLE_RATE);
  const timerRef = useRef<number | null>(null);
  const autoStopRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef(0);
  const recordingActiveRef = useRef(false);

  useEffect(() => {
    getConfig()
      .then((loaded) => {
        setConfig(loaded);
        setStatus("Python API connected");
      })
      .catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    return () => {
      cleanupRecording();
    };
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

  async function handleStartRecording() {
    setBusy(true);
    try {
      await startRecording();
      setStatus("Recording microphone");
    } catch (error) {
      cleanupRecording();
      setStatus(error instanceof Error ? error.message : "Microphone recording failed");
    } finally {
      setBusy(false);
    }
  }

  async function handleStopRecording() {
    if (!recordingActiveRef.current) return;
    setBusy(true);
    try {
      const audio = await stopRecording();
      if (!audio || audio.size === 0) {
        setStatus("No microphone audio captured");
        return;
      }
      setStatus("Transcribing with Yandex SpeechKit");
      const payload = await transcribeYandexSpeech(audio, sttLanguage, STT_SAMPLE_RATE);
      if (!payload.text) {
        setStatus("SpeechKit returned no text");
        return;
      }
      setTranscript((current) => joinTranscript(current, payload.text));
      setStatus("Speech transcript added");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Speech transcription failed");
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
          <div className="transcript-actions">
            <button className={recording ? "danger" : ""} onClick={recording ? handleStopRecording : handleStartRecording} disabled={busy}>
              {recording ? <Square size={16} /> : <Mic size={16} />}
              {recording ? `Stop ${recordingSeconds}s` : "Record mic"}
            </button>
            <select value={sttLanguage} onChange={(event) => setSttLanguage(event.target.value)} disabled={busy || recording}>
              <option value="ru-RU">Russian</option>
              <option value="en-US">English</option>
              <option value="tr-TR">Turkish</option>
            </select>
            <button onClick={handleDetect} disabled={busy}>Detect questions</button>
          </div>
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

  async function startRecording() {
    const audioContextCtor = window.AudioContext || getWebkitAudioContext();
    if (!audioContextCtor) {
      throw new Error("This browser does not support microphone recording");
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });
    const audioContext = new audioContextCtor({ sampleRate: STT_SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const gain = audioContext.createGain();
    gain.gain.value = 0;

    audioChunksRef.current = [];
    inputSampleRateRef.current = audioContext.sampleRate;
    recordingStartedAtRef.current = Date.now();
    processor.onaudioprocess = (event) => {
      audioChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };

    source.connect(processor);
    processor.connect(gain);
    gain.connect(audioContext.destination);

    audioContextRef.current = audioContext;
    sourceRef.current = source;
    processorRef.current = processor;
    gainRef.current = gain;
    streamRef.current = stream;
    recordingActiveRef.current = true;
    setRecordingSeconds(0);
    setRecording(true);
    timerRef.current = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordingStartedAtRef.current) / 1000);
      setRecordingSeconds(Math.min(elapsed, MAX_RECORDING_SECONDS));
    }, 250);
    autoStopRef.current = window.setTimeout(() => {
      void handleStopRecording();
    }, MAX_RECORDING_SECONDS * 1000);
  }

  async function stopRecording(): Promise<Blob | null> {
    const chunks = audioChunksRef.current;
    const inputSampleRate = inputSampleRateRef.current;
    cleanupRecording();
    if (chunks.length === 0) {
      return null;
    }
    const samples = mergeSamples(chunks);
    const resampled = inputSampleRate === STT_SAMPLE_RATE ? samples : resampleLinear(samples, inputSampleRate, STT_SAMPLE_RATE);
    return new Blob([encodePcm16(resampled)], { type: "application/octet-stream" });
  }

  function cleanupRecording() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (autoStopRef.current !== null) {
      window.clearTimeout(autoStopRef.current);
      autoStopRef.current = null;
    }
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    gainRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    void audioContextRef.current?.close();
    recordingActiveRef.current = false;
    audioContextRef.current = null;
    sourceRef.current = null;
    processorRef.current = null;
    gainRef.current = null;
    streamRef.current = null;
    setRecording(false);
    setRecordingSeconds(0);
  }
}

function getWebkitAudioContext(): typeof AudioContext | undefined {
  return (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

function joinTranscript(current: string, addition: string): string {
  const trimmedCurrent = current.trim();
  const trimmedAddition = addition.trim();
  if (!trimmedCurrent) return trimmedAddition;
  return `${trimmedCurrent}\n${trimmedAddition}`;
}

function mergeSamples(chunks: Float32Array[]): Float32Array {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function resampleLinear(samples: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (samples.length === 0 || sourceRate === targetRate) {
    return samples;
  }
  const ratio = sourceRate / targetRate;
  const newLength = Math.floor(samples.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i += 1) {
    const sourceIndex = i * ratio;
    const left = Math.floor(sourceIndex);
    const right = Math.min(left + 1, samples.length - 1);
    const weight = sourceIndex - left;
    result[i] = samples[left] * (1 - weight) + samples[right] * weight;
  }
  return result;
}

function encodePcm16(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}
