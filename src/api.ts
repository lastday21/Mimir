export type Provider = "yandex_ai_studio" | "ollama";
export type AudioMode = "yandex_realtime" | "speechkit";

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

export interface AudioDevice {
  id: string;
  name: string;
  source: "remote" | "mic";
  loopback: boolean;
  default: boolean;
  recommended: boolean;
}

export interface AudioDevicesPayload {
  available: boolean;
  devices: AudioDevice[];
  error?: string;
}

export interface LiveAudioSnapshot {
  running: boolean;
  mode?: AudioMode | "idle";
  sources: Array<"remote" | "mic">;
  language: string;
  sampleRateHertz: number;
  chunkDurationMs: number;
  vadEnabled: boolean;
  deviceIds?: Partial<Record<"remote" | "mic", string>>;
  tracePath?: string;
  lastError?: string;
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

export function listAudioDevices(): Promise<AudioDevicesPayload> {
  return request<AudioDevicesPayload>("/api/audio/devices");
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

export function startLiveAudio(
  sources: Array<"remote" | "mic">,
  deviceIds: Partial<Record<"remote" | "mic", string>> = {},
  mode: AudioMode = "yandex_realtime"
): Promise<LiveAudioSnapshot> {
  return request<LiveAudioSnapshot>("/api/session/audio/start", {
    method: "POST",
    body: JSON.stringify({ sources, deviceIds, language: "ru-RU", mode, vadEnabled: true })
  });
}

export function stopLiveAudio(): Promise<LiveAudioSnapshot> {
  return request<LiveAudioSnapshot>("/api/session/audio/stop", {
    method: "POST"
  });
}

export function sendTranscript(source: "remote" | "mic", text: string, isFinal = true): Promise<TranscriptTurn> {
  return request<TranscriptTurn>("/api/session/transcript", {
    method: "POST",
    body: JSON.stringify({ source, text, isFinal })
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
