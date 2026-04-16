import time
import sys
import os
from pathlib import Path
import yaml
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# Add risk_pipeline to sys.path so src can be found early
pipeline_root = Path(__file__).parent.resolve()
sys.path.append(str(pipeline_root))

# Diagnostic import
try:
    import lightgbm
    print(f"[API] Initial LightGBM check: OK (version {lightgbm.__version__})")
except ImportError as e:
    print(f"[API] Initial LightGBM check: FAILED: {e}")

from src.tfidf_lightgbm import TfidfLightGBM
from src.finbert        import FinBertModel
from src.fusion         import fuse_probs, risk_score, risk_level, harm_level, LABEL_MAP, apply_scam_heuristics
from src.explain        import generate_explanation

# ------------------------------------------------------------------
# 1. Lifespan & Model Loading
# ------------------------------------------------------------------
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Initializing Hybrid Risk Classifier Models...")

    config_path = pipeline_root / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    export_dir = pipeline_root / "exports"
    cfg["export_dir"] = str(export_dir)

    models['lgb']    = TfidfLightGBM.load(str(export_dir))
    models['finbert'] = FinBertModel.load(cfg)

    alpha_path = export_dir / "alpha.txt"
    models['alpha'] = float(alpha_path.read_text().strip())

    models['fuse_probs']      = fuse_probs
    models['risk_score']      = risk_score
    models['risk_level']      = risk_level
    models['harm_level']      = harm_level
    models['LABEL_MAP']       = LABEL_MAP
    models['apply_heuristics'] = apply_scam_heuristics

    print(f"[API] Models Loaded Successfully. Alpha={models['alpha']}")
    yield
    models.clear()


app = FastAPI(
    title="Hybrid Financial Risk API",
    description="AI-powered risk classification using Tuned LightGBM + FinBERT",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server and any localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# 2. Pydantic Models
# ------------------------------------------------------------------
class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    label: str
    label_id: int
    risk_score: float
    risk_level: str
    harm_level: str
    latency_ms: float
    heuristic_triggered: bool
    heuristic_type: str
    explanation: str
    recommendations: list[str]
    probabilities: dict

class IsSafeRequest(BaseModel):
    text: str

class IsSafeResponse(BaseModel):
    is_safe: bool
    label: str
    risk_score: float

class UrlRequest(BaseModel):
    url: str

class ImageRequest(BaseModel):
    image_base64: str   # base64-encoded image data
    mime_type: str = "image/jpeg"


# ------------------------------------------------------------------
# 3. Internal helper — run the ML pipeline on a text string
# ------------------------------------------------------------------
def _run_pipeline(text: str) -> dict:
    t_start = time.time()

    p_log = models['lgb'].predict_proba([text])
    p_fin = models['finbert'].predict_proba([text])
    final = models['fuse_probs'](p_log, p_fin, models['alpha'])

    final, triggered, h_type = models['apply_heuristics'](text, final)

    latency_ms = (time.time() - t_start) * 1000

    label_idx = int(final.argmax(axis=1)[0])
    r_score   = float(models['risk_score'](final)[0])
    label_text = models['LABEL_MAP'][label_idx]

    explanation, recommendations = generate_explanation(
        text,
        label_text,
        r_score,
        {k: round(float(final[0][i]), 4) for i, k in models['LABEL_MAP'].items()}
    )

    return {
        "label":               label_text,
        "label_id":            label_idx,
        "risk_score":          round(r_score, 4),
        "risk_level":          models['risk_level'](r_score),
        "harm_level":          models['harm_level'](label_idx),
        "latency_ms":          round(latency_ms, 1),
        "heuristic_triggered": triggered,
        "heuristic_type":      h_type,
        "explanation":         explanation,
        "recommendations":     recommendations,
        "probabilities": {
            "Safe":       round(float(final[0][0]), 4),
            "Misleading": round(float(final[0][1]), 4),
            "High Risk":  round(float(final[0][2]), 4),
            "Scam":       round(float(final[0][3]), 4),
        },
    }


# ------------------------------------------------------------------
# 4. Endpoints
# ------------------------------------------------------------------

@app.post("/v1/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        return _run_pipeline(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@app.post("/v1/is-safe", response_model=IsSafeResponse)
async def is_safe(request: IsSafeRequest):
    """
    Lightweight safety check used by the MCP chat agent to filter
    retrieved news content before showing it to the user.
    """
    if not request.text.strip():
        return IsSafeResponse(is_safe=True, label="Safe", risk_score=0.0)
    try:
        p_log  = models['lgb'].predict_proba([request.text])
        p_fin  = models['finbert'].predict_proba([request.text])
        final  = models['fuse_probs'](p_log, p_fin, models['alpha'])
        final, triggered, _ = models['apply_heuristics'](request.text, final)

        label_idx  = int(final.argmax(axis=1)[0])
        r_score    = float(models['risk_score'](final)[0])
        label_text = models['LABEL_MAP'][label_idx]

        return IsSafeResponse(
            # Allow Safe (0), Misleading (1), and High Risk (2) for chatbot purposes.
            # Only block blatant Scams (3).
            is_safe=label_idx in [0, 1, 2],   
            label=label_text,
            risk_score=round(r_score, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety Check Error: {str(e)}")


@app.post("/v1/classify-url", response_model=ClassificationResponse)
async def classify_url(request: UrlRequest):
    """
    Scrape a URL, extract readable text using Gemini, then classify.
    """
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    # Step 1: Fetch page HTML
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                request.url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; FinVerify/1.0)"}
            )
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {str(e)}")

    # Step 2: Use Gemini to extract the financial claim text from the HTML
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Extract the main financial claim, advice, or statement from the following HTML page. "
            "Return only the key financial text (1-4 sentences), no HTML tags, no filler.\n\n"
            f"HTML:\n{html[:8000]}"
        )
        extracted = gemini.generate_content(prompt).text.strip()
    except Exception as e:
        # Fallback: strip tags manually
        import re
        extracted = re.sub(r'<[^>]+>', ' ', html)
        extracted = ' '.join(extracted.split())[:2000]

    if not extracted:
        raise HTTPException(status_code=422, detail="Could not extract meaningful text from URL")

    try:
        return _run_pipeline(extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@app.post("/v1/classify-image", response_model=ClassificationResponse)
async def classify_image(request: ImageRequest):
    """
    Extract text from an image using Gemini Vision, then classify.
    """
    if not request.image_base64.strip():
        raise HTTPException(status_code=400, detail="Image data cannot be empty")

    try:
        import base64
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        gemini = genai.GenerativeModel("gemini-1.5-flash")

        image_bytes = base64.b64decode(request.image_base64)

        prompt = (
            "Extract any financial claim, advice, investment statement, or alert text visible in this image. "
            "Return only the financial text in plain sentences. If no financial text is present, say 'No financial content found.'"
        )
        response = gemini.generate_content([
            prompt,
            {"mime_type": request.mime_type, "data": image_bytes}
        ])
        extracted = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Vision Error: {str(e)}")

    if not extracted or extracted.lower().startswith("no financial"):
        raise HTTPException(status_code=422, detail="No financial content found in image")

    try:
        return _run_pipeline(extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": len(models) > 0}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
