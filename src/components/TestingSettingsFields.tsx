import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, Pause, Play, RefreshCw, Trash2 } from "lucide-react";
import {
  AppConfig,
  TestRecording,
  TestingSnapshot,
  deleteTestingRecording,
  getTestingSnapshot,
  startTestingReplay,
  stopTestingReplay
} from "../api";

interface TestingSettingsFieldsProps {
  busy: boolean;
  config: AppConfig;
  onConfigChange: (config: AppConfig) => void;
}

export function TestingSettingsFields({ busy, config, onConfigChange }: TestingSettingsFieldsProps) {
  const [snapshot, setSnapshot] = useState<TestingSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionId, setActionId] = useState("");
  const [error, setError] = useState("");

  const loadSnapshot = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const next = await getTestingSnapshot();
      setSnapshot(next);
      setError("");
    } catch (loadError) {
      setError(errorMessage(loadError, "Не удалось получить список записей"));
    } finally {
      if (showLoading) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSnapshot();
  }, [loadSnapshot]);

  const shouldPoll = Boolean(snapshot?.activeRecordingId || snapshot?.replay.state === "running");
  useEffect(() => {
    if (!shouldPoll) return;
    const timer = window.setInterval(() => void loadSnapshot(false), 1000);
    return () => window.clearInterval(timer);
  }, [loadSnapshot, shouldPoll]);

  const recordings = useMemo(
    () => [...(snapshot?.recordings ?? [])].sort((left, right) => right.startedAt.localeCompare(left.startedAt)),
    [snapshot?.recordings]
  );
  const replayRunning = snapshot?.replay.state === "running";

  async function handleReplay(recordingId: string) {
    setActionId(`replay:${recordingId}`);
    setError("");
    try {
      await startTestingReplay(recordingId);
      await loadSnapshot(false);
    } catch (replayError) {
      setError(errorMessage(replayError, "Не удалось запустить проверку"));
    } finally {
      setActionId("");
    }
  }

  async function handleStopReplay() {
    setActionId("stop");
    setError("");
    try {
      await stopTestingReplay();
      await loadSnapshot(false);
    } catch (stopError) {
      setError(errorMessage(stopError, "Не удалось остановить проверку"));
    } finally {
      setActionId("");
    }
  }

  async function handleDelete(recording: TestRecording) {
    if (!window.confirm(`Удалить запись от ${formatDate(recording.startedAt)}?`)) return;
    setActionId(`delete:${recording.id}`);
    setError("");
    try {
      const next = await deleteTestingRecording(recording.id);
      setSnapshot(next);
    } catch (deleteError) {
      setError(errorMessage(deleteError, "Не удалось удалить запись"));
    } finally {
      setActionId("");
    }
  }

  return (
    <section className="settings-section testing-settings">
      <div className="settings-subtitle settings-subtitle-compact">
        <h3>Тестирование</h3>
        <p>Записывайте реальные созвоны и повторяйте их позже без своего участия.</p>
      </div>

      <label className="testing-toggle">
        <input
          type="checkbox"
          checked={config.testing.enabled}
          disabled={busy}
          onChange={(event) => onConfigChange({
            ...config,
            testing: { enabled: event.target.checked }
          })}
        />
        <span>
          <strong>Записывать созвоны для проверки</strong>
          <small>Звук созвона и микрофон сохраняются отдельными дорожками.</small>
        </span>
      </label>
      <p className="setup-note">
        Запись начинается при включении прослушивания. Изменение применяется со следующего включения.
      </p>

      <div className="testing-recordings-head">
        <div>
          <h3>Записи</h3>
          <p>Повтор запускает распознавание, память и подготовку подсказок заново.</p>
        </div>
        <button type="button" onClick={() => void loadSnapshot()} disabled={loading || Boolean(actionId)}>
          <RefreshCw className={loading ? "spin" : ""} size={15} />
          Обновить
        </button>
      </div>

      {error && <p className="testing-message error">{error}</p>}
      {loading && !snapshot ? (
        <p className="testing-empty"><Loader2 className="spin" size={16} /> Загружаю записи…</p>
      ) : recordings.length === 0 ? (
        <p className="testing-empty">Записей пока нет.</p>
      ) : (
        <div className="testing-recordings">
          {recordings.map((recording) => {
            const isActive = snapshot?.activeRecordingId === recording.id || recording.status === "recording";
            const replayingThis = replayRunning && snapshot?.replay.recordingId === recording.id;
            const acting = actionId.endsWith(recording.id);
            return (
              <article className={`testing-recording ${replayingThis ? "active" : ""}`} key={recording.id}>
                <div className="testing-recording-main">
                  <strong>{formatDate(recording.startedAt)}</strong>
                  <span>{formatDuration(recording.durationMs)} · {trackLabel(recording)}</span>
                  <small className={`testing-recording-status ${recording.status}`}>
                    {recordingStatusLabel(recording)}
                  </small>
                </div>
                <div className="testing-recording-actions">
                  <button
                    type="button"
                    onClick={() => void handleReplay(recording.id)}
                    disabled={busy || Boolean(actionId) || replayRunning || isActive || recording.status === "failed"}
                  >
                    {actionId === `replay:${recording.id}` ? <Loader2 className="spin" size={15} /> : <Play size={15} />}
                    Повторить
                  </button>
                  <button
                    className="danger"
                    type="button"
                    title="Удалить запись"
                    aria-label={`Удалить запись от ${formatDate(recording.startedAt)}`}
                    onClick={() => void handleDelete(recording)}
                    disabled={busy || Boolean(actionId) || replayRunning || isActive}
                  >
                    {actionId === `delete:${recording.id}` ? <Loader2 className="spin" size={15} /> : <Trash2 size={15} />}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {snapshot && snapshot.replay.state !== "idle" && (
        <ReplayStatus
          snapshot={snapshot}
          stopping={actionId === "stop"}
          onStop={() => void handleStopReplay()}
        />
      )}
    </section>
  );
}

function ReplayStatus({
  snapshot,
  stopping,
  onStop
}: {
  snapshot: TestingSnapshot;
  stopping: boolean;
  onStop: () => void;
}) {
  const { replay } = snapshot;
  if (replay.state === "running") {
    const progress = replay.durationMs > 0 ? Math.min(100, replay.elapsedMs / replay.durationMs * 100) : 0;
    return (
      <section className="testing-replay active">
        <div className="testing-replay-head">
          <div>
            <strong>Идёт проверка</strong>
            <span>{formatDuration(replay.elapsedMs)} / {formatDuration(replay.durationMs)}</span>
          </div>
          <button type="button" onClick={onStop} disabled={stopping}>
            {stopping ? <Loader2 className="spin" size={15} /> : <Pause size={15} />}
            Остановить
          </button>
        </div>
        <div className="testing-progress"><span style={{ width: `${progress}%` }} /></div>
      </section>
    );
  }

  if (replay.state === "completed" && replay.report) {
    return (
      <section className="testing-replay completed">
        <strong>Проверка завершена</strong>
        <div className="testing-report">
          <span>Реплик созвона: {replay.report.remoteTurns}</span>
          <span>Ваших реплик: {replay.report.micTurns}</span>
          <span>Вопросов: {replay.report.questions}</span>
          <span>Подсказок: {replay.report.answers}</span>
          <span>Дублей: {replay.report.duplicates}</span>
          <span>Ошибок: {replay.report.errors}</span>
        </div>
      </section>
    );
  }

  return (
    <p className={`testing-message ${replay.state === "failed" ? "error" : ""}`}>
      {replay.state === "failed"
        ? replay.error || "Проверка завершилась с ошибкой."
        : "Проверка остановлена."}
    </p>
  );
}

function recordingStatusLabel(recording: TestRecording): string {
  if (recording.status === "recording") return "Записывается";
  if (recording.status === "failed") return recording.error || "Ошибка записи";
  if (recording.status === "incomplete") {
    if (!recording.tracks.remote && !recording.tracks.mic) return "Нет звуковых дорожек";
    if (!recording.tracks.remote) return "Нет звука созвона";
    if (!recording.tracks.mic) return "Нет звука микрофона";
    return "Неполная запись";
  }
  return "Готова";
}

function trackLabel(recording: TestRecording): string {
  const tracks: string[] = [];
  if (recording.tracks.remote) tracks.push("созвон");
  if (recording.tracks.mic) tracks.push("микрофон");
  return tracks.length > 0 ? tracks.join(" + ") : "без звука";
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function formatDuration(value: number): string {
  const totalSeconds = Math.max(0, Math.floor(value / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor(totalSeconds % 3600 / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}
