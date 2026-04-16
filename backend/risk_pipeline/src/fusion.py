import numpy as np
import re
from .utils import prob_sanity_check

# Label mappings
LABEL_MAP = {0: "Safe", 1: "Misleading", 2: "High Risk", 3: "Scam"}
HARM_MAP  = {0: "None", 1: "Low", 2: "Medium", 3: "Severe"}


def fuse_probs(
    p_logistic: np.ndarray,
    p_finbert:  np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Weighted average fusion:
        Final_Probs = alpha * P_logistic + (1 - alpha) * P_finbert

    Both inputs are validated before and after fusion.
    """
    prob_sanity_check(p_logistic, "Logistic")
    prob_sanity_check(p_finbert,  "FinBERT")

    fused = alpha * p_logistic + (1.0 - alpha) * p_finbert

    # Re-normalise to absorb floating-point drift
    fused = fused / fused.sum(axis=1, keepdims=True)
    prob_sanity_check(fused, "Fusion output")
    return fused


def risk_score(probs: np.ndarray) -> np.ndarray:
    """
    Risk_Score = 0*p0 + 1*p1 + 2*p2 + 3*p3
    Returns shape (n,) — one score per sample.
    """
    weights = np.array([0, 1, 2, 3], dtype=float)
    return (probs * weights).sum(axis=1)


def risk_level(score: float) -> str:
    """Map a scalar risk score [0, 3] to a risk level string."""
    if score < 0.75:
        return "Low"
    if score < 1.5:
        return "Moderate"
    if score < 2.25:
        return "High"
    return "Severe"


def harm_level(label_idx: int) -> str:
    return HARM_MAP.get(label_idx, "Unknown")


def apply_scam_heuristics(text: str, probs: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """
    Heuristics v2: The Financial Safety Shield.
    Detects specific fraud patterns that AI models often misclassify as 'Safe'.
    """
    text_clean = text.lower()
    new_probs = probs.copy()
    triggered = False
    h_type = "None"
    
    # --- 1. Pattern Detection (The "Safety Shield" Layer) ---
    
    # A. The Doubler (e.g. "Send 0.1 get 0.2")
    # More flexible spacing and word bridge
    doubler_pattern = r"(send|transfer|deposit|give).{1,50}(btc|eth|crypto|sol|bnb|usdt).{1,50}(get|receive|return|win|doubl).{1,50}(back|return|profit)"
    
    # B. The Airdrop (Phishing)
    # Detects high-value token/dollar claims
    airdrop_pattern = r"(airdrop|giveaway|reward|claim).{1,100}(\$[0-9,]{3,}|[0-9]{3,}\s+(tokens|coins|pepe|shib|doge))"
    
    # C. The Gatekeeper (Social Engineering)
    # Detects recruitment into high-gain groups (VIP, Insider, etc.)
    gatekeeper_pattern = r"(join|dm|message|group|exclusive).{1,100}(vip|private|insider|exclusive|secret|cash|signals|whatsapp|telegram)"

    # D. Extreme ROI (>1000% or "Guaranteed" + high %)
    # Only trigger if high percentage is linked to investment/profit context
    roi_percentage = re.search(r"([1-9][0-9]{2,})\s*%", text_clean)
    investment_context = any(w in text_clean for w in ["profit", "return", "yield", "apy", "invest", "bonus", "guaranteed"])
    extreme_roi = (roi_percentage is not None) and investment_context
    guaranteed_high = ("guaranteed" in text_clean or "no risk" in text_clean or "risk free" in text_clean) and re.search(r"[0-9]{1,}\s*%", text_clean)

    # --- 2. Dynamic Thresholding & Pivoting ---
    
    # SCAM_PROB is Label 3, MISLEADING is Label 1
    scam_prob = probs[0][3]
    misleading_prob = probs[0][1]
    suspicion = scam_prob + misleading_prob
    
    # Hard Fraud (Doubler/Airdrop/Extreme ROI) -> Needs only 10% suspicion
    if re.search(doubler_pattern, text_clean) or re.search(airdrop_pattern, text_clean) or extreme_roi:
        if suspicion > 0.10:
            triggered = True
            h_type = "Hard Fraud (Doubler/Airdrop/ROI)"
            
    # Soft Fraud (Gatekeeper/Guaranteed) -> Needs 20% suspicion
    elif re.search(gatekeeper_pattern, text_clean) or guaranteed_high:
        if suspicion > 0.20:
            triggered = True
            h_type = "Soft Fraud (Gatekeeper/ROI)"

    # --- 3. Final Pivot ---
    if triggered:
        # Pivot to Scam: Forced 90% confidence
        new_probs[0] = [0.01, 0.04, 0.05, 0.90]
        return new_probs, True, h_type
            
    return probs, False, "None"
