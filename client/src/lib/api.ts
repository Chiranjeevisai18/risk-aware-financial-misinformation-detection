// ─────────────────────────────────────────────────────────────
// Typed API client for FinVerify
// • Risk Pipeline   → http://localhost:8000  (text, url, image)
// • MCP Chat Agent  → http://localhost:9000  (chatbot)
// • Supabase        → persistence layer
// ─────────────────────────────────────────────────────────────

import { supabase } from "@/lib/supabase";

const RISK_API = process.env.NEXT_PUBLIC_RISK_API_URL  ?? "http://localhost:8000";
const CHAT_API = process.env.NEXT_PUBLIC_CHAT_API_URL  ?? "http://localhost:9000";

// ─────────────────────────────────────────────────────────────
// Session ID — one UUID per browser, persisted in localStorage
// ─────────────────────────────────────────────────────────────

export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();
  let id = localStorage.getItem("finverify_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("finverify_session_id", id);
  }
  return id;
}

export function resetSessionId(): string {
  const id = crypto.randomUUID();
  if (typeof window !== "undefined") {
    localStorage.setItem("finverify_session_id", id);
  }
  return id;
}

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export interface ClassifyResponse {
  label: string;
  label_id: number;
  risk_score: number;
  risk_level: string;
  harm_level: string;
  latency_ms: number;
  heuristic_triggered: boolean;
  heuristic_type: string;
  explanation: string;
  recommendations: string[];
  probabilities: {
    Safe: number;
    Misleading: number;
    "High Risk": number;
    Scam: number;
  };
}

export interface ArticleSummary {
  headline: string;
  source: string;
  date: string;
  link: string;
  summary: string;
  relevance: string;
}

export interface ChatResponse {
  type: "news" | "financial" | "general";
  query: string;
  answer: string;
  top_articles?: ArticleSummary[];
  risk_filtered?: boolean;
}

export interface RiskContext {
  label: string;
  risk_score: number;
  risk_level: string;
  harm_level: string;
  explanation: string;
}

// ─────────────────────────────────────────────────────────────
// Risk Pipeline
// ─────────────────────────────────────────────────────────────

async function riskFetch<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${RISK_API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Risk API error ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function classifyText(text: string): Promise<ClassifyResponse> {
  return riskFetch<ClassifyResponse>("/v1/classify", { text });
}

export async function classifyUrl(url: string): Promise<ClassifyResponse> {
  return riskFetch<ClassifyResponse>("/v1/classify-url", { url });
}

export async function classifyImage(base64: string, mimeType = "image/jpeg"): Promise<ClassifyResponse> {
  return riskFetch<ClassifyResponse>("/v1/classify-image", { image_base64: base64, mime_type: mimeType });
}

// ─────────────────────────────────────────────────────────────
// Supabase — ensure chat session row exists
// ─────────────────────────────────────────────────────────────

async function ensureChatSession(sessionId: string, userId: string): Promise<void> {
  const { data } = await supabase
    .from("chat_sessions")
    .select("id")
    .eq("id", sessionId)
    .maybeSingle();

  if (!data) {
    await supabase.from("chat_sessions").insert({
      id: sessionId,
      user_id: userId,
    });
  } else {
    // Update last_active
    await supabase
      .from("chat_sessions")
      .update({ last_active: new Date().toISOString() })
      .eq("id", sessionId);
  }
}

// ─────────────────────────────────────────────────────────────
// Supabase — save a chat message pair
// ─────────────────────────────────────────────────────────────

export async function saveChatMessages(
  sessionId: string,
  userId: string,
  userText: string,
  botText: string,
  articles?: ArticleSummary[],
  riskFiltered?: boolean,
): Promise<void> {
  try {
    await ensureChatSession(sessionId, userId);
    await supabase.from("chat_messages").insert([
      { session_id: sessionId, role: "user", text: userText },
      { session_id: sessionId, role: "bot",  text: botText, articles: articles ?? null, risk_filtered: riskFiltered ?? false },
    ]);
  } catch (e) {
    console.warn("[supabase] Failed to save chat messages:", e);
  }
}

// ─────────────────────────────────────────────────────────────
// Supabase — load chat history for a session
// ─────────────────────────────────────────────────────────────

export interface StoredMessage {
  role: "user" | "bot";
  text: string;
  articles?: ArticleSummary[];
  risk_filtered?: boolean;
  created_at: string;
}

export async function loadChatHistory(sessionId: string): Promise<StoredMessage[]> {
  try {
    const { data, error } = await supabase
      .from("chat_messages")
      .select("role, text, articles, risk_filtered, created_at")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: true });

    if (error || !data) return [];
    return data as StoredMessage[];
  } catch {
    return [];
  }
}

// ─────────────────────────────────────────────────────────────
// Supabase — save analysis result
// ─────────────────────────────────────────────────────────────

export async function saveAnalysisResult(
  userId: string,
  inputText: string,
  inputMode: "text" | "url" | "image",
  result: ClassifyResponse,
): Promise<void> {
  try {
    await supabase.from("analysis_results").insert({
      user_id:             userId,
      input_text:          inputText,
      input_mode:          inputMode,
      label:               result.label,
      risk_level:          result.risk_level,
      risk_score:          result.risk_score,
      harm_level:          result.harm_level,
      explanation:         result.explanation,
      recommendations:     result.recommendations,
      probabilities:       result.probabilities,
      heuristic_triggered: result.heuristic_triggered,
    });
  } catch (e) {
    console.warn("[supabase] Failed to save analysis result:", e);
  }
}

// ─────────────────────────────────────────────────────────────
// MCP Chat Agent
// ─────────────────────────────────────────────────────────────

export async function chatMessage(
  message: string,
  sessionId: string,
  context?: RiskContext,
): Promise<ChatResponse> {
  // Prepend RISK_CONTEXT to first message so the agent can use it
  let queryWithContext = message;
  if (context) {
    queryWithContext =
      `[RISK_CONTEXT]\n` +
      `Label: ${context.label}\n` +
      `Risk Level: ${context.risk_level}\n` +
      `Risk Score: ${context.risk_score.toFixed(2)}/3\n` +
      `Harm Level: ${context.harm_level}\n` +
      `Explanation: ${context.explanation}\n` +
      `[/RISK_CONTEXT]\n\n` +
      message;
  }

  const res = await fetch(`${CHAT_API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: queryWithContext, session_id: sessionId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Chat API error ${res.status}`);
  }
  return res.json() as Promise<ChatResponse>;
}
