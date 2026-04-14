import os
import re
from groq import Groq
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Use the smartest available model for reasoning
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are a Senior Financial Fraud Analyst and Risk Investigator. 
Your task is to analyze financial social media posts and explain the risk to users.

CONTEXT PROVIDED:
1. The post text.
2. Machine Learning labels (Safe, Misleading, High Risk, Scam).
3. Risk Score (0 to 3).

OUTPUT FORMAT:
Provide your response in exactly two sections:
1. EXPLANATION: A concise, 2-3 sentence human-readable explanation of why this post is risky or safe. Focus on linguistic red flags (e.g., urgency, guaranteed returns, lack of registration).
2. RECOMMENDATIONS: A bulleted list of 2-3 actionable safety steps for the user.

TONE:
Professional, objective, and urgent if a scam is detected.
"""

def generate_explanation(text: str, label: str, risk_score: float, probabilities: dict):
    """
    Calls Groq LLM to generate a human-readable explanation and recommendations.
    """
    prompt = f"""
    POST TEXT: "{text}"
    AI CLASSIFICATION: {label}
    RISK SCORE: {risk_score}/3
    PROBABILITIES: {probabilities}
    
    Explain the reasoning and provide recommendations.
    """
    
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        response_text = completion.choices[0].message.content
        
        # Robust parsing using Regex
        # Look for "EXPLANATION" and "RECOMMENDATIONS" regardless of numbering (1., 2., ##, etc.)
        exp_match = re.search(r"(?:EXPLANATION|1\.|##\s*EXPLANATION):?\s*(.*?)(?=(?:RECOMMENDATIONS|2\.|##\s*RECOMMENDATIONS)|$)", response_text, re.DOTALL | re.IGNORECASE)
        rec_match = re.search(r"(?:RECOMMENDATIONS|2\.|##\s*RECOMMENDATIONS):?\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        
        explanation = exp_match.group(1).strip() if exp_match else "Classification based on hybrid risk metrics."
        recommendations_str = rec_match.group(1).strip() if rec_match else "Consult a certified financial advisor."
        
        # Convert recommendations to a clean list
        # Splitting by common bullet points: *, -, •, or newline numbers
        recommendations = [r.strip("* -•").strip() for r in re.split(r"\n+[*\-•]?\s*", recommendations_str) if r.strip()]
        
        # Fallback if list generation failed
        if not recommendations:
            recommendations = ["Verify the investment with an official financial regulator."]
            
        return explanation, recommendations
    except Exception as e:
        print(f"[Groq Error] {e}")
        return "Explanation unavailable. Please exercise caution.", ["Consult an official financial regulator."]

if __name__ == "__main__":
    # Quick test
    test_text = "Invest now! 1000% returns guaranteed!"
    exp, rec = generate_explanation(test_text, "Scam", 2.8, {"Scam": 0.9})
    print(f"EXP: {exp}\nREC: {rec}")
