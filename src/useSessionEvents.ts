import { useCallback, useEffect, useRef, useState } from "react";
import { AudioMode, QuestionEvent, SessionSnapshot, TranscriptTurn } from "./api";

export type AudioSource = "remote" | "mic";

export interface AudioLevel {
  rms: number;
  level: number;
  speech: boolean;
}

export function useSessionEvents() {
  const [session, setSession] = useState<SessionSnapshot | null>(null);
  const [turns, setTurns] = useState<TranscriptTurn[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [audioLevels, setAudioLevels] = useState<Record<AudioSource, AudioLevel>>(emptyAudioLevels());
  const [audioMode, setAudioMode] = useState<AudioMode>("speechkit");
  const [audioRunning, setAudioRunning] = useState(false);
  const [status, setStatus] = useState("Подключение к Mimir");
  const activeQuestionId = useRef("");

  const applySessionSnapshot = useCallback((snapshot: SessionSnapshot) => {
    setSession(snapshot);
    setTurns(snapshot.memory.turns);
    const question = snapshot.currentQuestion;
    activeQuestionId.current = question?.questionId ?? "";
    setCurrentQuestion((current) => question?.question ?? (snapshot.state === "degraded" ? current : ""));
    setAnswer((current) => {
      if (question && snapshot.currentAnswer.questionId === question.questionId) {
        return snapshot.currentAnswer.text;
      }
      return snapshot.state === "degraded" ? current : "";
    });
  }, []);

  useEffect(() => {
    const events = new EventSource("/api/session/events");

    events.addEventListener("session_snapshot", (event) => {
      applySessionSnapshot(parseEvent<SessionSnapshot>(event));
    });

    events.addEventListener("session_state", (event) => {
      const payload = parseEvent<SessionSnapshot>(event);
      applySessionSnapshot(payload);
      const lastError = typeof payload.metrics.lastError === "string"
        ? runtimeErrorMessage(payload.metrics.lastError)
        : "";
      setStatus(payload.state === "degraded" ? lastError || "Сессия работает с ограничениями" : `Сессия: ${payload.state}`);
    });

    events.addEventListener("transcript", (event) => {
      const payload = parseEvent<TranscriptTurn>(event);
      setTurns((current) => mergeTranscriptTurn(current, payload));
    });

    events.addEventListener("question", (event) => {
      const payload = parseEvent<QuestionEvent>(event);
      activeQuestionId.current = payload.questionId;
      setCurrentQuestion(payload.question);
      setAnswer("");
      setStatus("Вопрос найден");
    });

    events.addEventListener("answer_delta", (event) => {
      const payload = parseEvent<{ questionId: string; deltaText: string }>(event);
      if (payload.questionId !== activeQuestionId.current) return;
      setAnswer((current) => current + payload.deltaText);
    });

    events.addEventListener("answer_done", (event) => {
      const payload = parseEvent<{ questionId: string }>(event);
      if (payload.questionId !== activeQuestionId.current) return;
      setStatus("Ответ готов");
    });

    events.addEventListener("answer_cancelled", (event) => {
      const payload = parseEvent<{ questionId: string }>(event);
      if (payload.questionId !== activeQuestionId.current) return;
      setStatus("Предыдущий ответ отменен");
    });

    events.addEventListener("answer_error", (event) => {
      const payload = parseEvent<{ questionId: string; question?: string; error: string }>(event);
      if (payload.questionId && payload.questionId !== activeQuestionId.current) return;
      const message = runtimeErrorMessage(payload.error);
      if (payload.question) {
        activeQuestionId.current = "";
        setCurrentQuestion(payload.question);
      }
      setAnswer(`Ответ не получен. ${message}`);
      setStatus(message);
    });

    events.addEventListener("transcript_uncertain", (event) => {
      const payload = parseEvent<{ message: string }>(event);
      setStatus(payload.message);
    });

    events.addEventListener("stt_status", (event) => {
      const payload = parseEvent<{ status: string; source: string }>(event);
      setStatus(`Распознавание ${payload.source}: ${payload.status}`);
    });

    events.addEventListener("stt_error", (event) => {
      const payload = parseEvent<{ error: string }>(event);
      setStatus(runtimeErrorMessage(payload.error));
    });

    events.addEventListener("stt_warning", (event) => {
      const payload = parseEvent<{ message: string }>(event);
      setStatus(payload.message);
    });

    events.addEventListener("stt_recovered", (event) => {
      const payload = parseEvent<{ message: string }>(event);
      setStatus(payload.message);
    });

    events.addEventListener("audio_status", (event) => {
      const payload = parseEvent<{ status: string; source?: string; running?: boolean; mode?: string }>(event);
      if (payload.mode && isAudioMode(payload.mode)) {
        setAudioMode(payload.mode);
      }
      if (typeof payload.running === "boolean") {
        setAudioRunning(payload.running);
        if (!payload.running) {
          setAudioLevels(emptyAudioLevels());
        }
      }
      const state = audioStatusLabel(payload.status);
      setStatus(payload.source ? `${audioSourceLabel(payload.source)}: ${state}` : `Звук: ${state}`);
    });

    events.addEventListener("audio_level", (event) => {
      const payload = parseEvent<{ source: AudioSource; rms: number; speech: boolean }>(event);
      if (!isAudioSource(payload.source)) return;
      setAudioLevels((current) => ({
        ...current,
        [payload.source]: {
          rms: payload.rms,
          level: rmsToLevel(payload.rms),
          speech: payload.speech
        }
      }));
    });

    events.addEventListener("audio_error", (event) => {
      const payload = parseEvent<{ error: string; running?: boolean }>(event);
      if (typeof payload.running === "boolean") {
        setAudioRunning(payload.running);
        if (!payload.running) {
          setAudioLevels(emptyAudioLevels());
        }
      }
      setStatus(runtimeErrorMessage(payload.error));
    });

    events.onerror = () => {
      setStatus("Нет связи с Mimir");
    };

    events.onopen = () => {
      setStatus((current) => current === "Нет связи с Mimir" ? "Mimir подключен" : current);
    };

    return () => events.close();
  }, [applySessionSnapshot]);

  return {
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
  };
}

function parseEvent<T>(event: Event): T {
  return JSON.parse((event as MessageEvent<string>).data) as T;
}

function mergeTranscriptTurn(current: TranscriptTurn[], update: TranscriptTurn): TranscriptTurn[] {
  const next = [...current];
  const index = next.findIndex((turn) => turn.turnId === update.turnId);
  if (index >= 0) {
    next[index] = update;
  } else {
    next.push(update);
  }
  const cutoff = update.timestampMs - (update.memoryWindowMs ?? 5 * 60 * 1000);
  return next.filter((turn) => turn.timestampMs >= cutoff);
}

export function emptyAudioLevels(): Record<AudioSource, AudioLevel> {
  return {
    remote: { rms: 0, level: 0, speech: false },
    mic: { rms: 0, level: 0, speech: false }
  };
}

function rmsToLevel(rms: number): number {
  return Math.max(0, Math.min(1, rms / 5000));
}

function isAudioSource(source: string): source is AudioSource {
  return source === "remote" || source === "mic";
}

export function isAudioMode(mode: string): mode is AudioMode {
  return mode === "yandex_realtime" || mode === "speechkit" || mode === "local_vosk";
}

function audioSourceLabel(source: string): string {
  if (source === "remote") return "Собеседник";
  if (source === "mic") return "Вы";
  return "Звук";
}

function audioStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    starting: "подключение",
    streaming: "звук поступает",
    speech: "идёт речь",
    silence: "пауза",
    fallback: "включён запасной способ распознавания",
    reconnecting: "проверка звука приложения",
    failed: "ошибка захвата",
    incomplete: "запись неполная",
    stopping: "остановка",
    stopped: "остановлено",
    done: "завершено",
    idle: "ожидание"
  };
  return labels[status] ?? status;
}

function runtimeErrorMessage(error: string): string {
  const normalized = error.toLowerCase();
  if (normalized.includes("localhost:11434") || normalized.includes("ollama")) {
    return "Локальная нейросеть не запущена.";
  }
  if (
    normalized.includes("speechkit") &&
    (normalized.includes("failed to connect") || normalized.includes("handshaker shutdown"))
  ) {
    return "Не удалось подключиться к распознаванию речи.";
  }
  if (normalized.includes("yandex ai studio failed")) {
    return "Не удалось получить ответ от Яндекса.";
  }
  return error;
}
