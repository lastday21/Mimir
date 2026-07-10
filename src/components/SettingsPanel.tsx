import { Check, Cloud, Loader2, Server } from "lucide-react";
import { AppConfig, AudioMode, ModelInfo, Provider } from "../api";

const PROVIDER_NAMES: Record<Provider, string> = {
  yandex_ai_studio: "Yandex AI Studio",
  ollama: "Ollama"
};

const AUDIO_MODE_NAMES: Record<AudioMode, string> = {
  yandex_realtime: "Яндекс Realtime — минимальная задержка",
  speechkit: "SpeechKit — запасной облачный путь",
  local_vosk: "Локально — Vosk и Ollama"
};

const SELECTABLE_AUDIO_MODES: AudioMode[] = ["yandex_realtime", "local_vosk"];

interface SettingsPanelProps {
  apiKey: string;
  busy: boolean;
  config: AppConfig;
  models: ModelInfo[];
  setupProblem: string | null;
  setupReady: boolean;
  onApiKeyChange: (value: string) => void;
  onAudioModeChange: (mode: AudioMode) => void;
  onConfigChange: (config: AppConfig) => void;
  onLoadModels: () => void;
  onProviderChange: (provider: Provider) => void;
  onSave: (closeAfterSave: boolean) => void;
}

export function SettingsPanel({
  apiKey,
  busy,
  config,
  models,
  setupProblem,
  setupReady,
  onApiKeyChange,
  onAudioModeChange,
  onConfigChange,
  onLoadModels,
  onProviderChange,
  onSave
}: SettingsPanelProps) {
  function updateHotkey(name: keyof AppConfig["hotkeys"], value: string) {
    onConfigChange({
      ...config,
      hotkeys: {
        ...config.hotkeys,
        [name]: value
      }
    });
  }

  return (
    <section className="panel settings-panel">
      <div className="section-title">
        {config.llmProvider === "ollama" ? <Server size={18} /> : <Cloud size={18} />}
        <h2>Настройка</h2>
      </div>

      <div className="settings-grid">
        <label>
          Провайдер
          <select value={config.llmProvider} onChange={(event) => onProviderChange(event.target.value as Provider)}>
            <option value="yandex_ai_studio">{PROVIDER_NAMES.yandex_ai_studio}</option>
            <option value="ollama">{PROVIDER_NAMES.ollama}</option>
          </select>
        </label>

        <label>
          Модель ответа
          <select
            value={config.llmModel}
            onChange={(event) => onConfigChange({ ...config, llmModel: event.target.value })}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {modelLabel(model)}
              </option>
            ))}
          </select>
        </label>

        <label>
          Обработка звука
          <select
            value={config.audioMode}
            onChange={(event) => onAudioModeChange(event.target.value as AudioMode)}
          >
            {SELECTABLE_AUDIO_MODES.map((mode) => (
              <option key={mode} value={mode}>{AUDIO_MODE_NAMES[mode]}</option>
            ))}
          </select>
        </label>
      </div>

      {config.llmProvider === "yandex_ai_studio" ? (
        <div className="provider-fields">
          <label>
            Папка Yandex Cloud
            <input
              value={config.yandexFolderId}
              onChange={(event) => onConfigChange({ ...config, yandexFolderId: event.target.value })}
              placeholder="folder id"
            />
          </label>

          <label>
            API-ключ Яндекса
            <input
              type="password"
              placeholder={config.hasYandexKey ? "Ключ сохранен" : "Вставьте API-ключ"}
              value={apiKey}
              onChange={(event) => onApiKeyChange(event.target.value)}
            />
          </label>
        </div>
      ) : (
        <div className="provider-fields single">
          <label>
            Адрес Ollama
            <input
              value={config.ollamaBaseUrl}
              onChange={(event) => onConfigChange({ ...config, ollamaBaseUrl: event.target.value })}
            />
          </label>
        </div>
      )}

      <div className="hotkey-fields">
        <label>
          Показать или скрыть окно подсказки
          <input
            value={config.hotkeys.overlayToggle}
            onChange={(event) => updateHotkey("overlayToggle", event.target.value)}
            placeholder="Ctrl+M"
          />
        </label>
        <label>
          Включить или поставить прослушивание на паузу
          <input
            value={config.hotkeys.audioToggle}
            onChange={(event) => updateHotkey("audioToggle", event.target.value)}
            placeholder="Ctrl+Space"
          />
        </label>
      </div>
      <p className="setup-note">Горячие клавиши применятся после перезапуска окна Mimir.</p>

      <div className="setup-actions">
        <button onClick={onLoadModels} disabled={busy}>
          <Loader2 className={busy ? "spin" : ""} size={16} />
          Обновить модели
        </button>
        <button onClick={() => onSave(false)} disabled={busy}>
          Сохранить
        </button>
        <button className="primary" onClick={() => onSave(true)} disabled={busy || !setupReady}>
          <Check size={16} />
          Открыть приложение
        </button>
      </div>

      {setupProblem && <p className="setup-hint">{setupProblem}</p>}
    </section>
  );
}

function modelLabel(model: ModelInfo): string {
  const context = model.contextWindow ? `, ${Math.round(model.contextWindow / 1000)}K` : "";
  return `${model.name || model.id}${context}`;
}
