import { useEffect, useMemo, useState } from "react";
import { Bot, Check, Loader2, MessageSquare, Pause, Play, Settings } from "lucide-react";
import {
  AppConfig,
  AudioMode,
  AudioDevice,
  ModelInfo,
  Provider,
  getConfig,
  listAudioDevices,
  listModels,
  pauseSession,
  preflightLiveAudio,
  saveConfig,
  startLiveAudio,
  storeYandexKey
} from "./api";
import { SettingsPanel } from "./components/SettingsPanel";
import {
  AudioLevel,
  AudioSource,
  emptyAudioLevels,
  isAudioMode,
  useSessionEvents
} from "./useSessionEvents";

const DEFAULT_CONFIG: AppConfig = {
  yandexFolderId: "",
  llmProvider: "yandex_ai_studio",
  llmModel: "yandexgpt/latest",
  audioMode: "yandex_realtime",
  ollamaBaseUrl: "http://localhost:11434",
  hasYandexKey: false,
  profile: {
    name: "",
    role: "",
    background: "",
    projects: "",
    stories: ""
  },
  conversation: {
    mode: "interview",
    goal: "",
    context: ""
  },
  setupCompleted: false,
  hotkeys: {
    overlayToggle: "Ctrl+M",
    audioToggle: "Ctrl+Space"
  }
};

const IS_OVERLAY = window.location.hash === "#overlay";

const DEFAULT_MODELS: Record<Provider, ModelInfo[]> = {
  yandex_ai_studio: [
    { id: "yandexgpt/latest", name: "YandexGPT", provider: "yandex_ai_studio", contextWindow: null },
    { id: "yandexgpt-lite/latest", name: "YandexGPT Lite", provider: "yandex_ai_studio", contextWindow: null }
  ],
  ollama: [
    { id: "qwen3:8b", name: "qwen3:8b", provider: "ollama", contextWindow: 32768 },
    { id: "qwen3:4b", name: "qwen3:4b", provider: "ollama", contextWindow: 32768 },
    { id: "qwen2.5:7b-instruct", name: "qwen2.5:7b-instruct", provider: "ollama", contextWindow: 32768 }
  ]
};

export function App() {
  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [apiKey, setApiKey] = useState("");
  const [modelsByProvider, setModelsByProvider] = useState<Record<Provider, ModelInfo[]>>(DEFAULT_MODELS);
  const [setupOpen, setSetupOpen] = useState(!IS_OVERLAY);
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([]);
  const [liveRemote, setLiveRemote] = useState(true);
  const [liveMic, setLiveMic] = useState(true);
  const [busy, setBusy] = useState(false);
  const {
    answer,
    applySessionSnapshot,
    audioLevels,
    audioMode,
    audioRunning,
    currentQuestion,
    session,
    setAudioLevels,
    setAudioMode,
    setAudioRunning,
    setStatus,
    status,
    turns
  } = useSessionEvents();

  useEffect(() => {
    document.body.classList.toggle("overlay-body", IS_OVERLAY);
    return () => document.body.classList.remove("overlay-body");
  }, []);

  useEffect(() => {
    getConfig()
      .then((loaded) => {
        setConfig(loaded);
        setAudioMode(loaded.audioMode);
        if (!IS_OVERLAY) {
          setSetupOpen(!loaded.setupCompleted);
        }
        setStatus("Сервер подключен");
      })
      .catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    listAudioDevices()
      .then((payload) => {
        if (payload.available) {
          setAudioDevices(payload.devices);
          return;
        }
        setAudioDevices([]);
        if (payload.error) {
          setStatus(payload.error);
        }
      })
      .catch((error) => setStatus(error.message));
  }, []);

  const providerModels = useMemo(() => {
    return modelOptionsForProvider(config, modelsByProvider);
  }, [config, modelsByProvider]);
  const setupProblem = setupValidationMessage(config, apiKey);
  const setupReady = setupProblem === null;

  async function persist(nextConfig = config) {
    let saved = await saveConfig(nextConfig);
    if (nextConfig.llmProvider === "yandex_ai_studio" && apiKey.trim()) {
      await storeYandexKey(apiKey.trim());
      setApiKey("");
      saved = await getConfig();
    }
    setConfig(saved);
    setStatus("Настройки сохранены");
    return saved;
  }

  function handleProviderChange(provider: Provider) {
    const nextMode: AudioMode = provider === "ollama"
      ? "local_vosk"
      : config.audioMode === "local_vosk"
        ? "yandex_realtime"
        : config.audioMode;
    setConfig({
      ...config,
      llmProvider: provider,
      llmModel: defaultModelForProvider(provider, modelsByProvider),
      audioMode: nextMode
    });
    setAudioMode(nextMode);
  }

  function handleAudioModeChange(mode: AudioMode) {
    const provider: Provider = mode === "local_vosk" ? "ollama" : "yandex_ai_studio";
    setConfig({
      ...config,
      audioMode: mode,
      llmProvider: provider,
      llmModel: provider === config.llmProvider
        ? config.llmModel
        : defaultModelForProvider(provider, modelsByProvider)
    });
    setAudioMode(mode);
  }

  async function handleSaveSetup(closeAfterSave: boolean) {
    const problem = setupValidationMessage(config, apiKey);
    if (problem) {
      setStatus(problem);
      return;
    }
    setBusy(true);
    try {
      const nextConfig = closeAfterSave ? { ...config, setupCompleted: true } : config;
      await persist(nextConfig);
      if (closeAfterSave) {
        setSetupOpen(false);
        setStatus("Настройки готовы");
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Не удалось сохранить настройки");
    } finally {
      setBusy(false);
    }
  }

  async function handleLoadModels() {
    setBusy(true);
    try {
      const saved = await persist();
      const payload = await listModels();
      const loadedModels = payload.models.filter((model) => model.provider === saved.llmProvider);
      const nextModels = loadedModels.length > 0 ? loadedModels : DEFAULT_MODELS[saved.llmProvider];
      setModelsByProvider((current) => ({
        ...current,
        [saved.llmProvider]: nextModels
      }));
      const preferredModel = payload.preferredModel || selectedOrFirstModel(saved.llmModel, nextModels);
      if (preferredModel && preferredModel !== saved.llmModel) {
        const next = { ...saved, llmModel: preferredModel };
        setConfig(next);
        await persist(next);
      }
      setStatus(`Моделей загружено: ${nextModels.length}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Не удалось загрузить модели");
    } finally {
      setBusy(false);
    }
  }

  async function handlePauseSession() {
    setBusy(true);
    try {
      const snapshot = await pauseSession();
      applySessionSnapshot(snapshot);
      setAudioRunning(false);
      setAudioLevels(emptyAudioLevels());
      setStatus("Сессия на паузе");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Не удалось поставить на паузу");
    } finally {
      setBusy(false);
    }
  }

  async function handleStartLiveAudio() {
    const sources: AudioSource[] = [];
    if (liveRemote) sources.push("remote");
    if (liveMic) sources.push("mic");
    if (sources.length === 0) return;
    setBusy(true);
    try {
      setAudioLevels(emptyAudioLevels());
      const deviceIds = recommendedDeviceIds(audioDevices, sources);
      const preflight = await preflightLiveAudio(sources, deviceIds, config.audioMode);
      if (!preflight.ok) {
        setStatus(preflight.errors[0] || "Проверка звука не прошла");
        return;
      }
      const snapshot = await startLiveAudio(sources, deviceIds, config.audioMode);
      if (snapshot.mode && isAudioMode(snapshot.mode)) {
        setAudioMode(snapshot.mode);
      }
      setAudioRunning(snapshot.running);
      setStatus(`Звук слушается: ${(snapshot.mode ?? config.audioMode).replace("_", " ")} ${snapshot.sources.join(" + ")}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Не удалось запустить звук");
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleLiveAudio() {
    if (audioRunning) {
      await handlePauseSession();
      return;
    }
    await handleStartLiveAudio();
  }

  const settingsPanel = (
    <SettingsPanel
      apiKey={apiKey}
      busy={busy}
      config={config}
      models={providerModels}
      setupProblem={setupProblem}
      setupReady={setupReady}
      onApiKeyChange={setApiKey}
      onAudioModeChange={handleAudioModeChange}
      onConfigChange={setConfig}
      onLoadModels={handleLoadModels}
      onProviderChange={handleProviderChange}
      onSave={handleSaveSetup}
    />
  );

  if (!IS_OVERLAY && setupOpen) {
    return (
      <main className="setup-main">
        <header className="app-header setup-header">
          <div>
            <h1>Mimir</h1>
            <p>Настройте цель разговора, свой профиль, модель и доступ.</p>
          </div>
          <span className="status">{busy ? <Loader2 className="spin" size={16} /> : <Check size={16} />} {status}</span>
        </header>
        {settingsPanel}
      </main>
    );
  }

  if (IS_OVERLAY) {
    return (
      <main className="overlay-shell">
        <header className="overlay-header">
          <span className={`overlay-dot ${audioRunning ? "active" : ""}`} />
          <span>{status}</span>
          {busy && <Loader2 className="spin" size={14} />}
        </header>

        <section className="overlay-question">
          <small>Вопрос</small>
          <p>{currentQuestion || "Жду вопрос собеседника."}</p>
        </section>

        <section className="overlay-answer">
          {answer || "Ответ появится здесь автоматически."}
        </section>

        <div className="overlay-levels">
          <AudioMeter compact label="Meet" deviceName={deviceLabel(audioDevices, "remote")} level={audioLevels.remote} />
          <AudioMeter compact label="Mic" deviceName={deviceLabel(audioDevices, "mic")} level={audioLevels.mic} />
        </div>

        <footer className="overlay-actions">
          <button onClick={handleToggleLiveAudio} disabled={busy}>
            {audioRunning ? <Pause size={15} /> : <Play size={15} />}
            {audioRunning ? "Пауза" : "Слушать"}
          </button>
          <span>{session?.state ?? "idle"}</span>
        </footer>
      </main>
    );
  }

  return (
    <main className="app-main">
      <header className="app-header">
        <div>
          <h1>Mimir</h1>
          <p>Помощник для живого собеседования и созвона.</p>
        </div>
        <div className="header-actions">
          <button className="icon-button" onClick={() => setSetupOpen(true)} disabled={busy} title="Настройки">
            <Settings size={18} />
          </button>
        </div>
      </header>

      <section className="status-bar">
        <StatusItem label="Приложение" value={audioRunning ? "включено" : "выключено"} active={audioRunning} />
        <StatusItem label="Нейронка" value={aiStatusLabel(config, audioRunning)} active={isAiConfigured(config)} />
        <StatusItem label="Разговор" value={conversationLabel(config)} active={Boolean(config.conversation.goal.trim())} />
        <StatusItem label="Звук" value={audioModeLabel(audioMode)} active={audioRunning} />
        <StatusItem label="Состояние" value={status} active={statusIsHealthy(status)} />
        <button className={audioRunning ? "listen-toggle danger" : "listen-toggle primary"} onClick={handleToggleLiveAudio} disabled={busy}>
          {busy ? <Loader2 className="spin" size={16} /> : audioRunning ? <Pause size={16} /> : <Play size={16} />}
          {audioRunning ? "Пауза" : "Включить"}
        </button>
      </section>

      <section className="live-layout">
        <section className="panel answer-panel">
          <div className="section-title">
            <Bot size={18} />
            <h2>Ответ</h2>
          </div>
          <article className="current-question">
            <small>Вопрос или тема</small>
            <p>{currentQuestion || "Жду вопрос собеседника."}</p>
          </article>
          <article className="answer-main">{answer || "Когда собеседник задаст вопрос, ответ появится здесь."}</article>
        </section>

        <section className="panel dialogue-panel">
          <div className="section-title">
            <MessageSquare size={18} />
            <h2>Диалог</h2>
          </div>
          <div className="messenger">
            {turns.length === 0 ? (
              <p className="dialogue-empty">После включения здесь появятся реплики встречи.</p>
            ) : (
              turns.map((turn) => (
                <div className={`message-bubble ${turn.source}`} key={turn.turnId}>
                  <strong>{turn.source === "remote" ? "Meet" : "Мы"}</strong>
                  <span>{turn.text}</span>
                </div>
              ))
            )}
          </div>
          <div className="dialogue-levels">
            <AudioMeter compact label="Meet" deviceName={deviceLabel(audioDevices, "remote")} level={audioLevels.remote} />
            <AudioMeter compact label="Мы" deviceName={deviceLabel(audioDevices, "mic")} level={audioLevels.mic} />
          </div>
        </section>
      </section>
    </main>
  );
}

function StatusItem({ active, label, value }: { active: boolean; label: string; value: string }) {
  return (
    <div className={`status-item ${active ? "active" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AudioMeter({
  compact = false,
  deviceName,
  label,
  level
}: {
  compact?: boolean;
  deviceName: string;
  label: string;
  level: AudioLevel;
}) {
  const percent = Math.round(level.level * 100);
  return (
    <div className={`audio-meter ${compact ? "compact" : ""} ${level.speech ? "speech" : ""}`}>
      <div className="audio-meter-head">
        <strong>{label}</strong>
        <span>{deviceName}</span>
      </div>
      <div className="audio-meter-track">
        <div className="audio-meter-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function recommendedDeviceIds(
  devices: AudioDevice[],
  sources: AudioSource[]
): Partial<Record<AudioSource, string>> {
  const ids: Partial<Record<AudioSource, string>> = {};
  for (const source of sources) {
    const device = preferredAudioDevice(devices, source);
    if (device) {
      ids[source] = device.id;
    }
  }
  return ids;
}

function deviceLabel(devices: AudioDevice[], source: AudioSource): string {
  return preferredAudioDevice(devices, source)?.name ?? (source === "remote" ? "Auto Meet output" : "Auto headset mic");
}

function preferredAudioDevice(devices: AudioDevice[], source: AudioSource): AudioDevice | undefined {
  return (
    devices.find((device) => device.source === source && device.recommended) ??
    devices.find((device) => device.source === source && device.default) ??
    devices.find((device) => device.source === source)
  );
}

function isAiConfigured(config: AppConfig): boolean {
  if (config.llmProvider === "ollama") {
    return Boolean(config.ollamaBaseUrl.trim());
  }
  return Boolean(config.hasYandexKey && config.yandexFolderId.trim());
}

function aiStatusLabel(config: AppConfig, audioRunning: boolean): string {
  if (audioRunning) {
    return config.llmProvider === "ollama" ? "локально работает" : "подключена";
  }
  if (!isAiConfigured(config)) {
    return "не настроена";
  }
  return config.llmProvider === "ollama" ? "локально готова" : "готова";
}

function audioModeLabel(mode: AudioMode): string {
  if (mode === "local_vosk") {
    return "локально";
  }
  if (mode === "speechkit") {
    return "SpeechKit, запасной";
  }
  return "Realtime API";
}

function conversationLabel(config: AppConfig): string {
  if (config.conversation.goal.trim()) {
    return config.conversation.goal.trim();
  }
  if (config.conversation.mode === "meeting") {
    return "обычная встреча";
  }
  if (config.conversation.mode === "technical") {
    return "техническое обсуждение";
  }
  if (config.conversation.mode === "custom") {
    return "своя цель";
  }
  return "собеседование";
}

function modelOptionsForProvider(
  config: AppConfig,
  modelsByProvider: Record<Provider, ModelInfo[]>
): ModelInfo[] {
  const options = modelsByProvider[config.llmProvider] ?? DEFAULT_MODELS[config.llmProvider];
  if (!config.llmModel.trim() || options.some((model) => model.id === config.llmModel)) {
    return options;
  }
  return [
    {
      id: config.llmModel,
      name: config.llmModel,
      provider: config.llmProvider,
      contextWindow: null
    },
    ...options
  ];
}

function defaultModelForProvider(provider: Provider, modelsByProvider: Record<Provider, ModelInfo[]>): string {
  return modelsByProvider[provider]?.[0]?.id ?? DEFAULT_MODELS[provider][0]?.id ?? "";
}

function selectedOrFirstModel(currentModel: string, models: ModelInfo[]): string {
  if (models.some((model) => model.id === currentModel)) {
    return currentModel;
  }
  return models[0]?.id ?? currentModel;
}

function setupValidationMessage(config: AppConfig, apiKey: string): string | null {
  if (config.conversation.mode === "custom" && !config.conversation.goal.trim()) {
    return "Опишите свою цель разговора.";
  }
  if (!config.llmModel.trim()) {
    return "Выберите модель.";
  }
  if (!config.hotkeys.overlayToggle.trim() || !config.hotkeys.audioToggle.trim()) {
    return "Укажите горячие клавиши.";
  }
  if (config.audioMode === "local_vosk" && config.llmProvider !== "ollama") {
    return "Для локального звука выберите Ollama.";
  }
  if (config.audioMode !== "local_vosk" && config.llmProvider !== "yandex_ai_studio") {
    return "Для облачного звука выберите Яндекс.";
  }
  if (config.llmProvider === "yandex_ai_studio" || config.audioMode !== "local_vosk") {
    if (!config.yandexFolderId.trim()) {
      return "Укажите папку Yandex Cloud.";
    }
    if (!config.hasYandexKey && !apiKey.trim()) {
      return "Укажите API-ключ Яндекса.";
    }
  }
  if (config.llmProvider === "ollama" && !config.ollamaBaseUrl.trim()) {
    return "Укажите адрес Ollama.";
  }
  return null;
}

function statusIsHealthy(status: string): boolean {
  const value = status.toLowerCase();
  return ![
    "ошибка",
    "нет связи",
    "error",
    "failed",
    "missing",
    "not found",
    "unavailable",
    "degraded",
    "ограничениями"
  ].some((marker) => value.includes(marker));
}
