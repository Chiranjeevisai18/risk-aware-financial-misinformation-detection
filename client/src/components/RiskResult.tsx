"use client";

import { motion } from "framer-motion";

type RiskLevel = "Low" | "Moderate" | "High" | "Severe";

interface RiskResultProps {
  riskScore: RiskLevel;           // risk_level string from backend
  riskScoreNum: number;           // numeric 0–3
  harmLevel: string;
  explanation: string;
  label: string;                  // e.g. "Scam", "Safe", "Misleading", "High Risk"
  keywords?: string[];
  probabilities: {
    Safe: number;
    Misleading: number;
    "High Risk": number;
    Scam: number;
  };
  recommendations: string[];
  heuristicTriggered: boolean;
  heuristicType: string;
  latencyMs: number;
}

const riskConfig = {
  Low: {
    color: "text-green-400",
    bg: "bg-green-400/10",
    border: "border-green-400/30",
    glow: "shadow-[0_0_30px_rgba(74,222,128,0.15)]",
    dot: "bg-green-400",
    bar: "bg-green-400",
    label: "LOW RISK",
  },
  Moderate: {
    color: "text-yellow-400",
    bg: "bg-yellow-400/10",
    border: "border-yellow-400/30",
    glow: "shadow-[0_0_30px_rgba(250,204,21,0.15)]",
    dot: "bg-yellow-400",
    bar: "bg-yellow-400",
    label: "MODERATE RISK",
  },
  High: {
    color: "text-orange-400",
    bg: "bg-orange-400/10",
    border: "border-orange-400/30",
    glow: "shadow-[0_0_30px_rgba(251,146,60,0.15)]",
    dot: "bg-orange-400",
    bar: "bg-orange-400",
    label: "HIGH RISK",
  },
  Severe: {
    color: "text-red-400",
    bg: "bg-red-400/10",
    border: "border-red-400/30",
    glow: "shadow-[0_0_30px_rgba(248,113,113,0.15)]",
    dot: "bg-red-400",
    bar: "bg-red-400",
    label: "SEVERE RISK",
  },
};

const probConfig: Record<string, { color: string; bg: string }> = {
  Safe:       { color: "bg-green-400",  bg: "bg-green-400/20" },
  Misleading: { color: "bg-yellow-400", bg: "bg-yellow-400/20" },
  "High Risk":{ color: "bg-orange-400", bg: "bg-orange-400/20" },
  Scam:       { color: "bg-red-400",    bg: "bg-red-400/20" },
};

export function RiskResult({
  riskScore,
  riskScoreNum,
  harmLevel,
  explanation,
  label,
  keywords = [],
  probabilities,
  recommendations,
  heuristicTriggered,
  heuristicType,
  latencyMs,
}: RiskResultProps) {
  const cfg = riskConfig[riskScore] ?? riskConfig["Severe"];
  const barWidth = `${Math.min(100, (riskScoreNum / 3) * 100).toFixed(1)}%`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="mt-6 flex flex-col gap-4"
    >
      {/* ── Heuristic Badge ── */}
      {heuristicTriggered && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center gap-2 bg-red-500/15 border border-red-500/30 rounded-xl px-4 py-2.5"
        >
          <span className="text-lg">🛡️</span>
          <div>
            <p className="text-red-400 text-xs font-bold uppercase tracking-widest" style={{ fontFamily: "'Sora', sans-serif" }}>
              Safety Shield Triggered
            </p>
            <p className="text-red-300/70 text-xs" style={{ fontFamily: "'Sora', sans-serif" }}>
              {heuristicType}
            </p>
          </div>
        </motion.div>
      )}

      {/* ── Row 1: Risk Score + Harm Level + Probabilities ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Risk Score Card */}
        <div className={`${cfg.bg} ${cfg.border} ${cfg.glow} border rounded-2xl p-6 flex flex-col items-center justify-center text-center`}>
          <span className="text-xs uppercase tracking-widest text-white/40 mb-3" style={{ fontFamily: "'Sora', sans-serif" }}>Risk Score</span>
          <div className={`w-3 h-3 rounded-full ${cfg.dot} mb-3 animate-pulse`} />
          <span className={`text-3xl font-bold ${cfg.color}`} style={{ fontFamily: "'Sora', sans-serif" }}>{riskScore}</span>
          <span className={`text-[10px] mt-1 ${cfg.color} opacity-60 font-mono`}>{cfg.label}</span>
          <span className="text-white/30 text-xs mt-1 font-mono">{label} ({riskScoreNum.toFixed(2)}/3)</span>
          <div className="w-full mt-4 bg-white/10 rounded-full h-1.5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: barWidth }}
              transition={{ duration: 1, ease: "easeOut" }}
              className={`${cfg.bar} h-1.5 rounded-full`}
            />
          </div>
        </div>

        {/* Harm Level Card */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 flex flex-col">
          <span className="text-xs uppercase tracking-widest text-white/40 mb-3" style={{ fontFamily: "'Sora', sans-serif" }}>Harm Level</span>
          <p className="text-white/80 text-sm leading-relaxed flex-1" style={{ fontFamily: "'Sora', sans-serif" }}>{harmLevel}</p>
        </div>

        {/* Probability Breakdown */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 flex flex-col gap-3">
          <span className="text-xs uppercase tracking-widest text-white/40 mb-1" style={{ fontFamily: "'Sora', sans-serif" }}>Model Probabilities</span>
          {Object.entries(probabilities).map(([key, val]) => {
            const pc = probConfig[key] ?? { color: "bg-blue-400", bg: "bg-blue-400/20" };
            const pct = `${(val * 100).toFixed(1)}%`;
            return (
              <div key={key} className="flex flex-col gap-1">
                <div className="flex justify-between text-xs" style={{ fontFamily: "'Sora', sans-serif" }}>
                  <span className="text-white/60">{key}</span>
                  <span className="text-white/50 font-mono">{pct}</span>
                </div>
                <div className={`w-full ${pc.bg} rounded-full h-1`}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: pct }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className={`${pc.color} h-1 rounded-full`}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Row 2: Explanation + Recommendations ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Explanation */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 flex flex-col">
          <span className="text-xs uppercase tracking-widest text-white/40 mb-3" style={{ fontFamily: "'Sora', sans-serif" }}>Explanation</span>
          <p className="text-white/80 text-sm leading-relaxed flex-1" style={{ fontFamily: "'Sora', sans-serif" }}>
            {explanation}
          </p>
        </div>

        {/* Recommendations */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 flex flex-col">
          <span className="text-xs uppercase tracking-widest text-white/40 mb-3" style={{ fontFamily: "'Sora', sans-serif" }}>Recommendations</span>
          <ul className="flex flex-col gap-2 flex-1">
            {recommendations.length > 0 ? recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-white/75" style={{ fontFamily: "'Sora', sans-serif" }}>
                <span className="text-blue-400 mt-0.5 shrink-0">→</span>
                {rec}
              </li>
            )) : (
              <li className="text-white/40 text-sm" style={{ fontFamily: "'Sora', sans-serif" }}>No specific recommendations.</li>
            )}
          </ul>
        </div>
      </div>

      {/* ── Latency badge ── */}
      <div className="flex justify-end">
        <span className="text-white/20 text-[10px] font-mono flex items-center gap-1">
          ⚡ {latencyMs}ms inference
        </span>
      </div>
    </motion.div>
  );
}