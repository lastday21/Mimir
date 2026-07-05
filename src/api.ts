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

export interface DetectedQuestion {
  text: string;
  confidence: number;
  timestampMs: number;
  source: string;
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

export function detectQuestions(text: string): Promise<{ questions: DetectedQuestion[] }> {
  return request<{ questions: DetectedQuestion[] }>("/api/detect", {
    method: "POST",
    body: JSON.stringify({ text })
  });
}

export function askMimir(question: string, transcript: string): Promise<{ answer: string }> {
  return request<{ answer: string }>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question, transcript })
  });
}
