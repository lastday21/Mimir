export type Provider = "yandex_ai_studio" | "ollama";

export interface AppConfig {
  yandexFolderId: string;
  llmProvider: Provider;
  llmModel: string;
  ollamaBaseUrl: string;
  hasYandexKey: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number | null;
}

export interface SessionSnapshot {
  sessionId: string;
  state: string;
  memory: {
    activeTopic: string;
    turns: TranscriptTurn[];
    questions: string[];
  };
  metrics: Record<string, unknown>;
}

export interface TranscriptTurn {
  source: "remote" | "mic";
  text: string;
  isFinal: boolean;
  timestampMs: number;
}

export interface QuestionEvent {
  sessionId: string;
  questionId: string;
  question: string;
  confidence: number;
  reason: string;
  context?: {
    activeTopic: string;
    priorQuestions: string[];
  };
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {})
    }
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data as T;
}

export function getConfig(): Promise<AppConfig> {
  return request<AppConfig>("/api/config");
}

export function saveConfig(config: AppConfig): Promise<AppConfig> {
  return request<AppConfig>("/api/config", {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export function storeYandexKey(apiKey: string): Promise<{ stored: true }> {
  return request<{ stored: true }>("/api/credentials/yandex", {
    method: "POST",
    body: JSON.stringify({ apiKey })
  });
}

export function listModels(): Promise<{ models: ModelInfo[]; preferredModel: string | null }> {
  return request<{ models: ModelInfo[]; preferredModel: string | null }>("/api/models");
}

export function startSession(): Promise<SessionSnapshot> {
  return request<SessionSnapshot>("/api/session/start", {
    method: "POST"
  });
}

export function stopSession(): Promise<SessionSnapshot> {
  return request<SessionSnapshot>("/api/session/stop", {
    method: "POST"
  });
}

export function sendTranscript(source: "remote" | "mic", text: string, isFinal = true): Promise<TranscriptTurn> {
  return request<TranscriptTurn>("/api/session/transcript", {
    method: "POST",
    body: JSON.stringify({ source, text, isFinal })
  });
}

export function askManualQuestion(question: string): Promise<QuestionEvent> {
  return request<QuestionEvent>("/api/manual/question", {
    method: "POST",
    body: JSON.stringify({ question })
  });
}

export async function uploadSpeechWav(
  source: "remote" | "mic",
  file: File,
  language = "ru-RU"
): Promise<{ started: true; jobId: string }> {
  const params = new URLSearchParams({ source, language });
  const response = await fetch(`/api/session/stt/wav?${params.toString()}`, {
    method: "POST",
    headers: {
      "Content-Type": "audio/wav"
    },
    body: file
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data as { started: true; jobId: string };
}
