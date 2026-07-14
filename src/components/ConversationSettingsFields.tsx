import { ConversationMode, ConversationSettings } from "../api";

const CONVERSATION_MODE_NAMES: Record<ConversationMode, string> = {
  interview: "Собеседование",
  meeting: "Обычная рабочая встреча",
  technical: "Техническое обсуждение",
  custom: "Своя цель"
};

interface ConversationSettingsFieldsProps {
  value: ConversationSettings;
  onChange: (value: ConversationSettings) => void;
}

export function ConversationSettingsFields({ value, onChange }: ConversationSettingsFieldsProps) {
  return (
    <section className="settings-section">
      <div className="settings-subtitle">
        <h3>Текущий разговор</h3>
        <p>Меняйте эти поля перед встречей — они определяют, когда и как Mimir должен помогать.</p>
      </div>

      <div className="context-settings-grid">
        <label>
          Сценарий
          <select
            value={value.mode}
            onChange={(event) => onChange({ ...value, mode: event.target.value as ConversationMode })}
          >
            {Object.entries(CONVERSATION_MODE_NAMES).map(([mode, name]) => (
              <option key={mode} value={mode}>{name}</option>
            ))}
          </select>
        </label>

        <label className="settings-wide-field">
          Цель разговора
          <textarea
            className="settings-textarea"
            value={value.goal}
            onChange={(event) => onChange({ ...value, goal: event.target.value })}
            placeholder="Например: понимать, что от меня требуется, и получать короткие варианты ответа"
          />
        </label>

        <label className="settings-wide-field">
          Контекст встречи
          <textarea
            className="settings-textarea"
            value={value.context}
            onChange={(event) => onChange({ ...value, context: event.target.value })}
            placeholder="Кто участвует, что обсуждается, какие решения уже приняты"
          />
        </label>
      </div>
    </section>
  );
}
