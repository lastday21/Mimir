import { UserProfile } from "../api";

interface UserProfileFieldsProps {
  value: UserProfile;
  onChange: (value: UserProfile) => void;
}

export function UserProfileFields({ value, onChange }: UserProfileFieldsProps) {
  return (
    <section className="settings-section">
      <div className="settings-subtitle">
        <h3>Профиль пользователя</h3>
        <p>Постоянные сведения помогают отвечать от вашего лица и не придумывать опыт.</p>
      </div>

      <div className="profile-settings-grid">
        <label>
          Имя
          <input
            value={value.name}
            onChange={(event) => onChange({ ...value, name: event.target.value })}
            placeholder="Как к вам обращаться"
          />
        </label>

        <label>
          Текущая роль
          <input
            value={value.role}
            onChange={(event) => onChange({ ...value, role: event.target.value })}
            placeholder="Например: разработчик Python"
          />
        </label>

        <label className="settings-wide-field">
          Опыт и резюме
          <textarea
            className="settings-textarea"
            value={value.background}
            onChange={(event) => onChange({ ...value, background: event.target.value })}
            placeholder="Ключевой опыт, навыки, сферы и факты, на которые можно опираться"
          />
        </label>

        <label className="settings-wide-field">
          Проекты
          <textarea
            className="settings-textarea"
            value={value.projects}
            onChange={(event) => onChange({ ...value, projects: event.target.value })}
            placeholder="Что делали, за что отвечали, какой получили результат"
          />
        </label>

        <label className="settings-wide-field">
          Подготовленные истории
          <textarea
            className="settings-textarea"
            value={value.stories}
            onChange={(event) => onChange({ ...value, stories: event.target.value })}
            placeholder="Сложные ситуации, решения, ошибки, достижения и примеры для ответа"
          />
        </label>
      </div>

      <p className="setup-note">
        Данные хранятся на этом компьютере. Для подготовки ответа нужный контекст передаётся выбранной модели.
      </p>
    </section>
  );
}
