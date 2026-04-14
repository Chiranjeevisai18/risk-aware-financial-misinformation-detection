"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send, Zap, Bot, AlertTriangle, ExternalLink, RefreshCw } from "lucide-react";
import {
  chatMessage,
  saveChatMessages,
  loadChatHistory,
  getOrCreateSessionId,
  resetSessionId,
  RiskContext,
  ArticleSummary,
} from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

interface Message {
  role: "user" | "bot";
  text: string;
  timestamp: Date;
  articles?: ArticleSummary[];
  riskFiltered?: boolean;
}

interface ChatBotProps {
  isOpen: boolean;
  onClose: () => void;
  riskContext?: RiskContext;
}

const QUICK_PROMPTS = [
  "What makes a claim High Risk?",
  "Latest market news?",
  "Explain my analysis result",
];

export function ChatBot({ isOpen, onClose, riskContext }: ChatBotProps) {
  const { user } = useAuth();

  const buildWelcome = (ctx?: RiskContext): Message => ({
    role: "bot",
    timestamp: new Date(),
    text: ctx
      ? `I've analyzed your text. It was classified as **${ctx.label}** with a risk level of **${ctx.risk_level}** (score: ${ctx.risk_score.toFixed(2)}/3).\n\n${ctx.explanation}\n\nAsk me anything about this claim or any financial topic.`
      : "Hi! I'm FinVerify's AI assistant powered by real-time financial data. Ask me about any financial claim, get market news, or explore risk analysis.",
  });

  const [messages, setMessages]   = useState<Message[]>([buildWelcome(riskContext)]);
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load or create session + reload history from Supabase on open
  useEffect(() => {
    if (!isOpen || !user) return;

    const sid = getOrCreateSessionId();
    setSessionId(sid);

    loadChatHistory(sid).then(history => {
      if (history.length === 0) {
        setMessages([buildWelcome(riskContext)]);
      } else {
        const restored: Message[] = history.map(m => ({
          role:        m.role,
          text:        m.text,
          timestamp:   new Date(m.created_at),
          articles:    m.articles,
          riskFiltered: m.risk_filtered,
        }));
        setMessages(restored);
      }
      setHistoryLoaded(true);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, user?.id]);

  // Inject riskContext banner whenever a new analysis result comes in
  // This always runs — even if history is loaded from Supabase
  const prevContextRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (!riskContext) return;
    const key = `${riskContext.label}-${riskContext.risk_score}`;
    if (prevContextRef.current === key) return; // same result, skip
    prevContextRef.current = key;

    const contextBanner: Message = {
      role: "bot",
      timestamp: new Date(),
      text: `I've analyzed your text. It was classified as **${riskContext.label}** with a risk level of **${riskContext.risk_level}** (score: ${riskContext.risk_score.toFixed(2)}/3).\n\n${riskContext.explanation}\n\nAsk me anything about this claim or any financial topic.`,
    };
    // Append context banner to current history (only if not already present)
    setMessages(prev => {
      // Check if we already have this specific analysis banner in the last 2 messages
      const isAlreadyShown = prev.some(m => m.text?.includes(riskContext.explanation.substring(0, 20)));
      if (isAlreadyShown) return prev;

      if (prev.length <= 1) {
        return [contextBanner]; // Replace initial welcome with detailed context
      }
      return [...prev, contextBanner]; // Append to ongoing conversation
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [riskContext]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleNewSession = () => {
    const newSid = resetSessionId();
    setSessionId(newSid);
    setHistoryLoaded(false);
    setMessages([buildWelcome(riskContext)]);
  };

  const sendMessage = async (text?: string) => {
    const trimmed = (text || input).trim();
    if (!trimmed || loading) return;

    const userMsg: Message = { role: "user", text: trimmed, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // Only pass riskContext on the first user message in this session
    const isFirstUserMsg = !messages.some(m => m.role === "user");
    const ctxToSend = isFirstUserMsg ? riskContext : undefined;

    try {
      const response = await chatMessage(trimmed, sessionId, ctxToSend);

      const botMsg: Message = {
        role:        "bot",
        text:        response.answer ?? "(No response received)",
        timestamp:   new Date(),
        articles:    response.top_articles,
        riskFiltered: response.risk_filtered,
      };
      setMessages(prev => [...prev, botMsg]);

      // Persist to Supabase
      if (user) {
        saveChatMessages(
          sessionId,
          user.id,
          trimmed,
          botMsg.text,
          botMsg.articles,
          botMsg.riskFiltered,
        );
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Something went wrong.";
      setMessages(prev => [
        ...prev,
        {
          role: "bot",
          text: `⚠️ Error: ${errMsg}. Please make sure the chat service is running on port 9000.`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (d: Date) =>
    d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const renderBotText = (text: string | undefined | null) => {
    if (!text) return null;
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i} className="text-white font-semibold">{part.slice(2, -2)}</strong>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 30, scale: 0.92 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 30, scale: 0.92 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
          className="fixed bottom-6 right-6 z-50 w-[92vw] max-w-[420px] flex flex-col overflow-hidden rounded-3xl"
          style={{
            maxHeight: "80vh",
            background: "linear-gradient(135deg, rgba(8,8,24,0.97) 0%, rgba(12,12,28,0.97) 100%)",
            border: "1px solid rgba(96,165,250,0.2)",
            boxShadow: "0 32px 80px rgba(2,2,12,0.9), 0 0 0 1px rgba(96,165,250,0.05), inset 0 1px 0 rgba(255,255,255,0.06)",
            backdropFilter: "blur(20px)",
          }}
        >
          {/* Lightning accent top border */}
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-400/60 to-transparent" />

          {/* Header */}
          <div
            className="flex items-center justify-between px-5 py-4 relative overflow-hidden shrink-0"
            style={{ background: "linear-gradient(135deg, rgba(37,99,235,0.15) 0%, rgba(109,40,217,0.12) 100%)", borderBottom: "1px solid rgba(96,165,250,0.12)" }}
          >
            <svg className="absolute right-16 top-0 h-full w-20 opacity-20 pointer-events-none" viewBox="0 0 80 60" xmlns="http://www.w3.org/2000/svg">
              <polyline points="60,0 50,25 58,25 42,60" fill="none" stroke="rgba(147,197,253,0.8)" strokeWidth="1.5" strokeLinecap="round" />
            </svg>

            <div className="flex items-center gap-3 relative z-10">
              <div className="relative w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: "linear-gradient(135deg, #2563eb, #7c3aed)" }}>
                <Zap className="w-4 h-4 text-white" />
                <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-green-400 rounded-full border-2 border-[#08081a] animate-pulse" />
              </div>
              <div>
                <p className="text-white text-sm font-bold leading-tight" style={{ fontFamily: "'Sora', sans-serif" }}>FinVerify AI</p>
                <p className="text-blue-300/60 text-xs" style={{ fontFamily: "'Sora', sans-serif" }}>MCP + Risk Filter · Online</p>
              </div>
            </div>

            <div className="flex items-center gap-2 relative z-10">
              {/* New session button */}
              <button
                onClick={handleNewSession}
                title="Start new chat session"
                className="w-8 h-8 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/10 transition-all"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/10 transition-all"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4 min-h-0">
            {/* Quick prompts (only at start with no loaded history) */}
            {messages.length === 1 && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap gap-2 mt-1">
                {QUICK_PROMPTS.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(p)}
                    className="text-xs px-3 py-1.5 rounded-full border border-blue-400/25 bg-blue-400/8 text-blue-300/80 hover:bg-blue-400/15 hover:text-blue-200 transition-all"
                    style={{ fontFamily: "'Sora', sans-serif" }}
                  >
                    {p}
                  </button>
                ))}
              </motion.div>
            )}

            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.3 }}
                className={`flex flex-col gap-1 ${msg.role === "user" ? "items-end" : "items-start"}`}
              >
                {msg.role === "bot" && (
                  <div className="flex items-start gap-2 w-full">
                    <div className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0 mt-1" style={{ background: "linear-gradient(135deg, #2563eb, #7c3aed)" }}>
                      <Bot className="w-3.5 h-3.5 text-white" />
                    </div>
                    <div className="flex flex-col gap-2 flex-1 min-w-0">
                      {msg.riskFiltered && (
                        <div className="flex items-center gap-2 bg-orange-500/15 border border-orange-500/30 rounded-xl px-3 py-2">
                          <AlertTriangle className="w-3.5 h-3.5 text-orange-400 shrink-0" />
                          <p className="text-orange-300/80 text-xs" style={{ fontFamily: "'Sora', sans-serif" }}>
                            Some retrieved content was flagged as risky and filtered out.
                          </p>
                        </div>
                      )}
                      <div
                        className="max-w-[90%] px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-relaxed text-white/85"
                        style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", fontFamily: "'Sora', sans-serif", whiteSpace: "pre-wrap" }}
                      >
                        {renderBotText(msg.text)}
                      </div>
                      {msg.articles && msg.articles.length > 0 && (
                        <div className="flex flex-col gap-2 w-full">
                          {msg.articles.slice(0, 3).map((article, ai) => (
                            <a
                              key={ai}
                              href={article.link !== "N/A" ? article.link : "#"}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block bg-white/4 border border-white/8 rounded-xl px-3 py-2.5 hover:bg-white/8 transition-all group"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <p className="text-white/80 text-xs font-medium leading-snug line-clamp-2" style={{ fontFamily: "'Sora', sans-serif" }}>{article.headline}</p>
                                  <p className="text-white/35 text-[10px] mt-1" style={{ fontFamily: "'Sora', sans-serif" }}>{article.source} · {article.date}</p>
                                  {article.summary && (
                                    <p className="text-white/45 text-[10px] mt-1 line-clamp-2" style={{ fontFamily: "'Sora', sans-serif" }}>{article.summary}</p>
                                  )}
                                </div>
                                {article.link !== "N/A" && (
                                  <ExternalLink className="w-3.5 h-3.5 text-white/30 group-hover:text-blue-400 transition-colors shrink-0 mt-0.5" />
                                )}
                              </div>
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {msg.role === "user" && (
                  <div
                    className="max-w-[85%] px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed text-white"
                    style={{ background: "linear-gradient(135deg, #2563eb, #7c3aed)", boxShadow: "0 4px 20px rgba(99,102,241,0.3)", fontFamily: "'Sora', sans-serif" }}
                  >
                    {msg.text}
                  </div>
                )}

                <span className="text-white/20 text-[10px] px-1" style={{ fontFamily: "'Sora', sans-serif" }}>
                  {formatTime(msg.timestamp)}
                </span>
              </motion.div>
            ))}

            {loading && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex items-end gap-2">
                <div className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0" style={{ background: "linear-gradient(135deg, #2563eb, #7c3aed)" }}>
                  <Bot className="w-3.5 h-3.5 text-white" />
                </div>
                <div className="px-4 py-3 rounded-2xl rounded-bl-sm flex items-center gap-1.5" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                  {[0, 1, 2].map(i => (
                    <div key={i} className="w-2 h-2 rounded-full bg-blue-400/60 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                  ))}
                </div>
              </motion.div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div
            className="px-4 py-3 flex gap-2 relative shrink-0"
            style={{ borderTop: "1px solid rgba(96,165,250,0.1)", background: "rgba(0,0,0,0.2)" }}
          >
            <div className="flex-1 flex items-center gap-2 rounded-2xl px-4 py-2.5 text-sm" style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendMessage()}
                placeholder="Ask about a financial claim or market news..."
                className="flex-1 bg-transparent text-white placeholder:text-white/25 focus:outline-none text-sm"
                style={{ fontFamily: "'Sora', sans-serif" }}
              />
            </div>
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="w-10 h-10 rounded-xl flex items-center justify-center text-white transition-all disabled:opacity-30 disabled:cursor-not-allowed shrink-0"
              style={{ background: "linear-gradient(135deg, #2563eb, #7c3aed)", boxShadow: input.trim() ? "0 4px 15px rgba(99,102,241,0.4)" : "none" }}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>

          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple-400/40 to-transparent" />
        </motion.div>
      )}
    </AnimatePresence>
  );
}