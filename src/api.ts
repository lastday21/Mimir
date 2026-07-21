export type Provider = "yandex_ai_studio" | "ollama";
export type AudioMode = "yandex_realtime" | "speechkit" | "local_vosk";
export type ConversationMode = "interview" | "meeting" | "technical" | "custom";

export interface UserProfile {
  name: string;
  role: string;
  background: string;
  projects: string;
  stories: string;
}

export interface ConversationSettings {
  mode: ConversationMode;
  goal: string;
  context: string;
}

export interface TestingSettings {
  enabled: boolean;
}

export interface AppConfig {
  yandexFolderId: string;
  llmProvider: Provider;
  llmModel: string;
  audioMode: AudioMode;
  audioApplication: AudioApplicationSelection;
  ollamaBaseUrl: string;
  hasYandexKey: boolean;
  profile: UserProfile;
  conversation: ConversationSettings;
  testing: TestingSettings;
  setupCompleted: boolean;
  hotkeys: {
    overlayToggle: string;
    audioToggle: string;
  };
}

export interface AudioApplicationSelection {
  processId: number;
  executable: string;
  title: string;
}

export type AudioApplication = AudioApplicationSelection;

export interface AudioApplicationsPayload {
  available: boolean;
  applications: AudioApplication[];
  error?: string;
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
  eventSequence: number;
  memory: {
    activeTopic: string;
    windowMs: number;
    turns: TranscriptTurn[];
    questions: string[];
    exchanges: DialogueExchange[];
  };
  metrics: Record<string, unknown>;
  currentQuestion: QuestionEvent | null;
  currentAnswer: {
    questionId: string;
    text: string;
  };
}

export interface TranscriptTurn {
  turnId: string;
  source: "remote" | "mic";
  text: string;
  isFinal: boolean;
  uncertain?: boolean;
  startedAtMs: number;
  timestampMs: number;
  operation?: "append" | "replace";
  memoryWindowMs?: number;
}

export interface DialogueExchange {
  questionId: string;
  question: string;
  hint: string;
  userAnswer: string;
  timestampMs: number;
  updatedAtMs: number;
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
  applicationProcessId?: number;
  tracePath?: string;
  lastError?: string;
}

export interface AudioPreflightCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface AudioPreflightResult {
  ok: boolean;
  mode: AudioMode;
  sources: Array<"remote" | "mic">;
  deviceIds: Partial<Record<"remote" | "mic", string>>;
  applicationProcessId: number;
  checks: AudioPreflightCheck[];
  errors: string[];
}

export type TestRecordingStatus = "recording" | "ready" | "incomplete" | "failed";
export type TestReplayStatus = "idle" | "running" | "completed" | "stopped" | "failed";

export interface TestRecording {
  id: string;
  startedAt: string;
  durationMs: number;
  status: TestRecordingStatus;
  tracks: {
    remote: boolean;
    mic: boolean;
  };
  sizeBytes: number;
  error?: string;
}

export interface TestReplayReport {
  remoteTurns: number;
  micTurns: number;
  questions: number;
  answers: number;
  duplicates: number;
  errors: number;
}

export interface TestReplay {
  state: TestReplayStatus;
  recordingId: string | null;
  elapsedMs: number;
  durationMs: number;
  error?: string;
  report?: TestReplayReport;
}

export interface TestingSnapshot {
  activeRecordingId: string | null;
  recordings: TestRecording[];
  replay: TestReplay;
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

export function listAudioApplications(): Promise<AudioApplicationsPayload> {
  return request<AudioApplicationsPayload>("/api/audio/applications");
}

export function getTestingSnapshot(): Promise<TestingSnapshot> {
  return request<TestingSnapshot>("/api/testing");
}

export function startTestingReplay(recordingId: string): Promise<TestReplay> {
  return request<TestReplay>("/api/testing/replay/start", {
    method: "POST",
    body: JSON.stringify({ recordingId })
  });
}

export function stopTestingReplay(): Promise<TestReplay> {
  return request<TestReplay>("/api/testing/replay/stop", {
    method: "POST"
  });
}

export function deleteTestingRecording(recordingId: string): Promise<TestingSnapshot> {
  return request<TestingSnapshot>("/api/testing/recordings/delete", {
    method: "POST",
    body: JSON.stringify({ recordingId })
  });
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

export function pauseSession(): Promise<SessionSnapshot> {
  return request<SessionSnapshot>("/api/session/pause", {
    method: "POST"
  });
}

export function startLiveAudio(
  sources: Array<"remote" | "mic">,
  deviceIds: Partial<Record<"remote" | "mic", string>> = {},
  mode: AudioMode = "speechkit",
  applicationProcessId = 0
): Promise<LiveAudioSnapshot> {
  return request<LiveAudioSnapshot>("/api/session/audio/start", {
    method: "POST",
    body: JSON.stringify({ sources, deviceIds, applicationProcessId, language: "ru-RU", mode, vadEnabled: true })
  });
}

export function preflightLiveAudio(
  sources: Array<"remote" | "mic">,
  deviceIds: Partial<Record<"remote" | "mic", string>> = {},
  mode: AudioMode = "speechkit",
  applicationProcessId = 0
): Promise<AudioPreflightResult> {
  return request<AudioPreflightResult>("/api/session/audio/preflight", {
    method: "POST",
    body: JSON.stringify({ sources, deviceIds, applicationProcessId, language: "ru-RU", mode, vadEnabled: true })
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
