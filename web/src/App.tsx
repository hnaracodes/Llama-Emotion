import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChatResponse,
  fetchHealth,
  HealthResponse,
  resetSession,
  sendChat,
  Traits,
} from "./api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  tone?: string;
  collapse?: boolean;
  recovered?: boolean;
}

function sessionId(): string {
  const key = "llama-emotion-session";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

function toneClass(tone?: string): string {
  if (!tone) return "tone-pill";
  return `tone-pill ${tone}`;
}

function TraitBars({ traits }: { traits?: Traits }) {
  if (!traits) return <p style={{ color: "var(--muted)", fontSize: "0.8rem" }}>—</p>;
  const keys = ["engagement", "arousal", "warmth", "tension", "shift"] as const;
  return (
    <div>
      {keys.map((k) => (
        <div key={k} className="metric-row">
          <span>{k}</span>
          <span>{(traits[k] ?? 0).toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const sid = useMemo(() => sessionId(), []);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMeta, setLastMeta] = useState<ChatResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [collapseBanner, setCollapseBanner] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setError(null);
    setCollapseBanner(null);
    setMessages((m) => [
      ...m,
      { id: crypto.randomUUID(), role: "user", content: text },
    ]);
    setLoading(true);

    try {
      const res = await sendChat(text, sid);
      setLastMeta(res);
      if (res.collapse_detected) {
        setCollapseBanner(
          res.recovered
            ? "Collapse guard: affect-modulated reply collapsed; recovered with hooks disabled."
            : "Collapse guard: returned a safe fallback response.",
        );
      }
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: res.reply,
          tone: res.dominant_tone,
          collapse: res.collapse_detected,
          recovered: res.recovered,
        },
      ]);
      void refreshHealth();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function onReset() {
    if (!confirm("Clear chat and reset affect state?")) return;
    await resetSession(sid);
    setMessages([]);
    setLastMeta(null);
    setCollapseBanner(null);
    setError(null);
    void refreshHealth();
  }

  return (
    <div className="app-shell">
      <section className="chat-panel">
        <header className="header">
          <h1>Llama-Emotion</h1>
          <p>
            W4 Llama + affective gate · encoder → SNN → hooks · v3.1 hardened
          </p>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-hint">
              Share what&apos;s on your mind. Replies are modulated by a
              32-d affect vector — tone and traits appear in the sidebar.
            </p>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`bubble ${m.role}`}>
              {m.role === "assistant" && m.tone && (
                <div className="bubble-meta">
                  <span className={toneClass(m.tone)}>{m.tone}</span>
                  {m.collapse && (
                    <span style={{ marginLeft: 8, color: "var(--warm)" }}>
                      {m.recovered ? "· guard recovered" : "· guard fallback"}
                    </span>
                  )}
                </div>
              )}
              {m.content}
            </div>
          ))}
          {loading && <div className="typing">Amygdala is thinking…</div>}
          <div ref={bottomRef} />
        </div>

        {collapseBanner && (
          <div className="collapse-banner">{collapseBanner}</div>
        )}
        {error && (
          <div
            className="collapse-banner"
            style={{
              borderColor: "rgba(248,113,113,0.4)",
              background: "rgba(248,113,113,0.1)",
              color: "var(--alert)",
            }}
          >
            {error}
            <br />
            <small>
              Start API: <code>py -3 run_microscope.py</code> or deploy Modal.
            </small>
          </div>
        )}

        <form className="composer" onSubmit={onSubmit}>
          <textarea
            rows={1}
            placeholder="Type a message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void onSubmit(e);
              }
            }}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </section>

      <aside className="sidebar">
        <div className="card">
          <h2>Gate health</h2>
          {health?.gate ? (
            <>
              <p className={health.gate.healthy ? "gate-ok" : "gate-warn"}>
                {health.gate.healthy ? "OK" : "Warning"} · {health.gate.source}
              </p>
              <div className="metric-row">
                <span>version</span>
                <span>{health.gate.version ?? "?"}</span>
              </div>
              {health.gate.warning && (
                <p style={{ fontSize: "0.72rem", color: "var(--warm)", marginTop: 8 }}>
                  {health.gate.warning}
                </p>
              )}
            </>
          ) : (
            <p style={{ fontSize: "0.8rem", color: "var(--muted)" }}>
              Connect API to see gate status
            </p>
          )}
        </div>

        <div className="card">
          <h2>Current tone</h2>
          <span className={toneClass(lastMeta?.dominant_tone)}>
            {lastMeta?.dominant_tone ?? "neutral"}
          </span>
        </div>

        <div className="card">
          <h2>Affect traits</h2>
          <TraitBars traits={lastMeta?.traits} />
        </div>

        {lastMeta?.introspection && (
          <div className="card">
            <h2>Introspection</h2>
            <div className="metric-row">
              <span>hooks_on</span>
              <span>{String(lastMeta.introspection.hooks_on)}</span>
            </div>
            <div className="metric-row">
              <span>affect_norm</span>
              <span>
                {(lastMeta.introspection.affect_vector_norm ?? 0).toFixed(3)}
              </span>
            </div>
            <div className="metric-row">
              <span>rolling_kl</span>
              <span>
                {(lastMeta.introspection.rolling_kl_vs_hooks_off ?? 0).toFixed(4)}
              </span>
            </div>
          </div>
        )}

        <div className="sidebar-actions">
          <button type="button" onClick={() => void refreshHealth()}>
            Refresh status
          </button>
          <button type="button" onClick={() => void onReset()}>
            Reset chat
          </button>
        </div>
      </aside>
    </div>
  );
}
