import os
import asyncio
from datetime import datetime, timedelta, UTC
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the new FastMCP client
from src.stock_market_client import StockMarketClient

load_dotenv()

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

RISK_API_URL = os.getenv("RISK_API_URL", "http://localhost:8000")

async def _is_safe(text: str) -> bool:
    if not text or not text.strip():
        return True
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.post(
                f"{RISK_API_URL}/v1/is-safe",
                json={"text": text[:2000]},
            )
            data = resp.json()
            return data.get("is_safe", True)
    except httpx.TimeoutException:
        print("[risk-filter] ⚠️ Risk server timed out (2.5s) — failing open.")
        return True
    except Exception as exc:
        print(f"[risk-filter] ⚠️ Risk service unreachable: {exc} — failing open.")
        return True


async def _filter_response(result: dict) -> dict:
    risk_filtered = False

    if result.get("type") == "general" or result.get("answer", "").startswith("⚠️"):
        return result

    answer = result.get("answer", "")
    import re
    first_sentence = re.split(r'[.!?]\s', answer)[0].strip() if answer else ""
    if first_sentence and not await _is_safe(first_sentence):
        result["answer"] = (
            "⚠️ The generated response was flagged for safety and has been condensed. "
            "Some information has been withheld. Please consult a verified "
            "financial source (SEBI/RBI) for final investment advice."
        )
        risk_filtered = True

    articles = result.get("top_articles", [])
    if articles:
        safe_articles = []
        for article in articles:
            check_text = article.get('headline', '')
            if await _is_safe(check_text):
                safe_articles.append(article)
            else:
                risk_filtered = True
                print(f"[risk-filter] Blocked article: {article.get('headline', '')}")
        result["top_articles"] = safe_articles

    result["risk_filtered"] = risk_filtered
    return result


sessions: dict[str, tuple[StockMarketClient, datetime]] = {}
SESSION_TTL_MINUTES = 30

async def _cleanup_idle_sessions():
    while True:
        await asyncio.sleep(300)
        cutoff = datetime.now(UTC) - timedelta(minutes=SESSION_TTL_MINUTES)
        expired = [sid for sid, (_, last) in sessions.items() if last < cutoff]
        for sid in expired:
            agent, _ = sessions[sid]
            await agent.cleanup()
            del sessions[sid]
            print(f"[sessions] Expired session: {sid}")
        if expired:
            print(f"[sessions] Cleaned up {len(expired)} idle session(s). Active: {len(sessions)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("━" * 52)
    print("  📈 Stock Market Agent API | FastMCP + Web Search  ")
    print("━" * 52)
    
    cleanup_task = asyncio.create_task(_cleanup_idle_sessions())

    # Pre-warm a default session
    default_agent = StockMarketClient(session_id="__default__")
    await default_agent.connect_to_server("./src/stock_market_server.py")
    sessions["__default__"] = (default_agent, datetime.now(UTC))
    print(f"✅ Application is fully ready.")

    yield

    cleanup_task.cancel()
    print("🛑 Shutting down server...")
    for sid, (agent, _) in sessions.items():
        await agent.cleanup()

app = FastAPI(
    title="Stock Market Agent API",
    description="FastMCP + Groq + Web Search Client Wrapper",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    query: str
    session_id: str = "__default__"
    reset: bool = False

async def _run_agent(req: NewsRequest) -> dict:
    sid = req.session_id or "__default__"

    if req.reset and sid in sessions:
        agent, _ = sessions[sid]
        await agent.cleanup()
        del sessions[sid]

    if sid not in sessions:
        new_agent = StockMarketClient(session_id=sid)
        await new_agent.connect_to_server("./src/stock_market_server.py")
        await new_agent.load_history()
        sessions[sid] = (new_agent, datetime.now(UTC))
        print(f"[sessions] New session: {sid}. Active sessions: {len(sessions)}")
    else:
        agent_inst, _ = sessions[sid]
        sessions[sid] = (agent_inst, datetime.now(UTC))

    agent_inst, _ = sessions[sid]

    try:
        # process_query returns {"type": "news", ...} now
        result = await agent_inst.process_query(req.query)
        
        # If it returned a raw string by accident somehow, format it
        if isinstance(result, str):
             result = {"type": "general", "query": req.query, "answer": result, "top_articles": []}
             
        result = await _filter_response(result)
        return result
    except Exception as exc:
        print(f"[server] ❌ Unexpected Error: {exc}")
        return {
            "type": "general",
            "query": req.query,
            "answer": f"⚠️ Service is temporarily unavailable. Please try again later. (Details: {str(exc)[:100]})",
            "top_articles": [],
            "error": str(exc)
        }

@app.post("/api/news")
async def api_news(req: NewsRequest):
    return await _run_agent(req)

@app.post("/api/chat")
async def api_chat(req: NewsRequest):
    return await _run_agent(req)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
