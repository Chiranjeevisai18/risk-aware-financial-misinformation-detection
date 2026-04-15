"""
Financial Agent — LangGraph + Groq + MCP Server
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

from components.mcp_manager import MCPToolManager
from components.agent import FinancialAgent

load_dotenv()

# --- SSL CERTIFICATE FIX ---
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
print(f"[SSL] Certificates linked successfully from {certifi.where()}")

RISK_API_URL = os.getenv("RISK_API_URL", "http://localhost:8000")


async def _is_safe(text: str) -> bool:
    """
    Calls the risk_pipeline /v1/is-safe endpoint.
    Fail-open: if risk service is unreachable or slow, content passes through.
    """
    if not text or not text.strip():
        return True
    try:
        # Reduced timeout to 2.5s to prevent UI hangs if the risk server is down
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
    """
    Filters the synthesized answer and each article through the risk pipeline.
    Removes unsafe content and sets risk_filtered=True if anything was removed.
    """
    risk_filtered = False

    answer = result.get("answer", "")
    if answer and not await _is_safe(answer):
        result["answer"] = (
            "⚠️ The retrieved information was flagged as potentially misleading "
            "and has been withheld for your safety. Please consult a verified "
            "financial source such as SEBI, RBI, or an authorised advisor."
        )
        risk_filtered = True

    articles = result.get("top_articles", [])
    if articles:
        safe_articles = []
        for article in articles:
            check_text = f"{article.get('headline', '')} {article.get('summary', '')}"
            if await _is_safe(check_text):
                safe_articles.append(article)
            else:
                risk_filtered = True
                print(f"[risk-filter] Blocked article: {article.get('headline', '')}")
        result["top_articles"] = safe_articles

    result["risk_filtered"] = risk_filtered
    return result

# ─────────────────────────────────────────────
# FASTAPI SERVER ENTRY POINT
# ─────────────────────────────────────────────

import asyncio
from datetime import datetime, timedelta, UTC

mcp_managers: list[MCPToolManager] = []

# Per-session agent store: { session_id -> (agent, last_active) }
sessions: dict[str, tuple[FinancialAgent, datetime]] = {}
SESSION_TTL_MINUTES = 30


async def _cleanup_idle_sessions():
    """Periodically remove sessions idle for more than SESSION_TTL_MINUTES."""
    while True:
        await asyncio.sleep(300)  # run every 5 minutes
        cutoff = datetime.now(UTC) - timedelta(minutes=SESSION_TTL_MINUTES)
        expired = [sid for sid, (_, last) in sessions.items() if last < cutoff]
        for sid in expired:
            del sessions[sid]
            print(f"[sessions] Expired session: {sid}")
        if expired:
            print(f"[sessions] Cleaned up {len(expired)} idle session(s). Active: {len(sessions)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_managers
    print("━" * 52)
    print("  📈 Financial Agent  |  LangGraph + Groq + MCP Server")
    print("━" * 52)

    mcp_managers.append(MCPToolManager(server_script="./stock_market_server.py"))

    av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if av_api_key:
        mcp_managers.append(MCPToolManager(
            command="uvx",
            args=["--from", "marketdata-mcp-server", "marketdata-mcp", av_api_key]
        ))
    else:
        print("⚠️ ALPHA_VANTAGE_API_KEY not found. Skipping Alpha Vantage tools.")

    for mgr in mcp_managers:
        await mgr.__aenter__()

    # Start idle session cleanup background task
    cleanup_task = asyncio.create_task(_cleanup_idle_sessions())

    # Pre-warm a default session so the first request is fast
    default_agent = FinancialAgent(mcp_managers)
    sessions["__default__"] = (default_agent, datetime.now(UTC))
    print(f"✅ Application is fully ready. Total tools loaded: {len(default_agent.tools)}")

    yield

    cleanup_task.cancel()
    print("🛑 Shutting down server...")
    for mgr in reversed(mcp_managers):
        await mgr.__aexit__(None, None, None)

app = FastAPI(
    title="Financial Agent API",
    description="LangGraph + Groq + MCP with Risk Filter",
    version="1.0.0",
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
    session_id: str = "__default__"   # UUID from frontend localStorage
    reset: bool = False


async def _run_agent(req: NewsRequest) -> dict:
    sid = req.session_id or "__default__"

    if req.reset and sid in sessions:
        del sessions[sid]

    if sid not in sessions:
        new_agent = FinancialAgent(mcp_managers, session_id=sid)
        # Load persistent history from Supabase if available
        await new_agent.load_history()
        sessions[sid] = (new_agent, datetime.now(UTC))
        print(f"[sessions] New session: {sid}. Active sessions: {len(sessions)}")
    else:
        agent_inst, _ = sessions[sid]
        sessions[sid] = (agent_inst, datetime.now(UTC))  # refresh timestamp

    agent_inst, _ = sessions[sid]

    try:
        result = await agent_inst.chat(req.query)
        result = await _filter_response(result)
        return result
    except Exception as exc:
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/news")
async def api_news(req: NewsRequest):
    return await _run_agent(req)


@app.post("/api/chat")
async def api_chat(req: NewsRequest):
    """ChatBot endpoint — same pipeline as /api/news with risk filter applied."""
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