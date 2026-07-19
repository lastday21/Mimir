import { useState } from "react";
import { Check, Cloud, Server } from "lucide-react";
import { AppConfig, AudioApplication, AudioMode, ModelInfo, Provider } from "../api";
import { ConversationSettingsFields } from "./ConversationSettingsFields";
import { ModelSettingsFields } from "./ModelSettingsFields";
import { TestingSettingsFields } from "./TestingSettingsFields";
import { UserProfileFields } from "./UserProfileFields";

type SettingsSection = "conversation" | "profile" | "model" | "testing";

interface SettingsPanelProps {
  apiKey: string;
  busy: boolean;
  config: AppConfig;
  audioApplications: AudioApplication[];
  models: ModelInfo[];
  setupProblem: string | null;
  setupReady: boolean;
  onApiKeyChange: (value: string) => void;
  onAudioModeChange: (mode: AudioMode) => void;
  onConfigChange: (config: AppConfig) => void;
  onLoadModels: () => void;
  onLoadAudioApplications: () => void;
  onProviderChange: (provider: Provider) => void;
  onSave: (closeAfterSave: boolean) => void;
}

export function SettingsPanel({
  apiKey,
  busy,
  config,
  audioApplications,
  models,
  setupProblem,
  setupReady,
  onApiKeyChange,
  onAudioModeChange,
  onConfigChange,
  onLoadModels,
  onLoadAudioApplications,
  onProviderChange,
  onSave
}: SettingsPanelProps) {
  const [activeSection, setActiveSection] = useState<SettingsSection>("conversation");

  return (
    <section className="panel settings-panel">
      <div className="section-title">
        {config.llmProvider === "ollama" ? <Server size={18} /> : <Cloud size={18} />}
        <h2>Настройка</h2>
      </div>

      <nav className="settings-tabs" aria-label="Разделы настроек">
        <button
          className={activeSection === "conversation" ? "active" : ""}
          onClick={() => setActiveSection("conversation")}
        >
          Разговор
        </button>
        <button
          className={activeSection === "profile" ? "active" : ""}
          onClick={() => setActiveSection("profile")}
        >
          Профиль
        </button>
        <button
          className={activeSection === "model" ? "active" : ""}
          onClick={() => setActiveSection("model")}
        >
          Модель и звук
        </button>
        <button
          className={activeSection === "testing" ? "active" : ""}
          onClick={() => setActiveSection("testing")}
        >
          Тестирование
        </button>
      </nav>

      {activeSection === "conversation" && (
        <ConversationSettingsFields
          value={config.conversation}
          onChange={(conversation) => onConfigChange({ ...config, conversation })}
        />
      )}

      {activeSection === "profile" && (
        <UserProfileFields
          value={config.profile}
          onChange={(profile) => onConfigChange({ ...config, profile })}
        />
      )}

      {activeSection === "model" && (
        <ModelSettingsFields
          apiKey={apiKey}
          busy={busy}
          config={config}
          audioApplications={audioApplications}
          models={models}
          onApiKeyChange={onApiKeyChange}
          onAudioModeChange={onAudioModeChange}
          onConfigChange={onConfigChange}
          onLoadModels={onLoadModels}
          onLoadAudioApplications={onLoadAudioApplications}
          onProviderChange={onProviderChange}
        />
      )}

      {activeSection === "testing" && (
        <TestingSettingsFields
          busy={busy}
          config={config}
          onConfigChange={onConfigChange}
        />
      )}

      <div className="setup-actions">
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
