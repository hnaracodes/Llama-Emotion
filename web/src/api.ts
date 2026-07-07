const base = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type Traits = Record<string, number>;

export interface ChatResponse {
  reply: string;
  dominant_tone?: string;
  traits?: Traits;
  source?: string;
  gate_version?: string;
  gate_healthy?: boolean;
  collapse_detected?: boolean;
  collapse_score?: number;
  recovered?: boolean;
  introspection?: {
    hooks_on?: boolean;
    affect_vector_norm?: number;
    rolling_kl_vs_hooks_off?: number;
  };
}

export interface HealthResponse {
  ok: boolean;
  gate?: {
    source?: string;
    version?: string;
    expected_version?: string;
    healthy?: boolean;
    warning?: string | null;
  };
}

function apiUrl(path: string): string {
  return `${base}${path}`;
}

export async function sendChat(
  message: string,
  sessionId: string,
): Promise<ChatResponse> {
  const res = await fetch(apiUrl("/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<ChatResponse>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(apiUrl("/health"));
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json() as Promise<HealthResponse>;
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch(apiUrl(`/reset/${sessionId}`), { method: "POST" });
}
