"""
Trade Opportunities API
FastAPI service that analyzes market data and provides trade opportunity
insights for specific sectors in India.
Uses Groq API (free, fast, no quota issues).
"""

import time
import uuid
import logging
from datetime import datetime
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("trade_api")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = "llama-3.3-70b-versatile"

VALID_API_KEYS: dict[str, str] = {
    "demo-key-001": "demo_user_1",
    "demo-key-002": "demo_user_2",
    "guest-key-000": "guest",
}

RATE_LIMIT_REQUESTS: int = 5
RATE_LIMIT_WINDOW_SECONDS: int = 60
CACHE_TTL_SECONDS: int = 600

rate_limit_store: dict[str, list[float]] = defaultdict(list)
session_store: dict[str, dict] = {}
report_cache: dict[str, dict] = {}

VALID_SECTORS: set[str] = {
    "pharmaceuticals", "technology", "agriculture", "textiles",
    "automobiles", "energy", "finance", "healthcare", "manufacturing",
    "chemicals", "metals", "real_estate", "fmcg", "infrastructure",
    "it_services", "telecom", "retail", "defence", "aviation", "gems_jewellery",
}

app = FastAPI(
    title="Trade Opportunities API",
    description="Analyzes market data and provides structured trade opportunity insights for specific sectors in India.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    user_id = VALID_API_KEYS.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return user_id

def check_rate_limit(user_id: str = Depends(get_current_user)) -> str:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    rate_limit_store[user_id] = [t for t in rate_limit_store[user_id] if t > window_start]
    if len(rate_limit_store[user_id]) >= RATE_LIMIT_REQUESTS:
        oldest = rate_limit_store[user_id][0]
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - oldest)) + 1
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry after {retry_after}s.", headers={"Retry-After": str(retry_after)})
    rate_limit_store[user_id].append(now)
    return user_id

def get_or_create_session(request: Request, user_id: str) -> str:
    session_id = request.headers.get("X-Session-ID")
    if session_id and session_id in session_store:
        session_store[session_id]["requests"] += 1
        session_store[session_id]["last_seen"] = datetime.utcnow().isoformat()
    else:
        session_id = str(uuid.uuid4())
        session_store[session_id] = {"session_id": session_id, "user_id": user_id, "created": datetime.utcnow().isoformat(), "last_seen": datetime.utcnow().isoformat(), "requests": 1}
    return session_id

async def analyse_with_groq(sector: str) -> str:
    prompt = f"""You are a senior trade analyst specialising in Indian markets.
Sector under analysis: **{sector.upper()}**
Today's date: {datetime.utcnow().strftime('%B %d, %Y')}
Produce a structured Markdown trade-opportunity report with exactly these sections:
# Trade Opportunity Report - {sector.title()} Sector (India)
## 1. Executive Summary
## 2. Sector Overview
## 3. Current Market Trends
## 4. Trade Opportunities
(Table with columns: Opportunity | Target Markets | Estimated Value | Confidence)
## 5. Export Opportunities
## 6. Import Opportunities
## 7. Risks & Challenges
## 8. Policy & Regulatory Landscape
## 9. Recommended Actions
## 10. Data Sources & Disclaimer
Be specific to India. Use professional language."""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a senior trade analyst specialising in Indian markets. Always respond in structured Markdown format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 4096,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(GROQ_URL, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"})
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq API returned {resp.status_code}. Check your API key.")
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")

@app.get("/analyze/{sector}", response_class=PlainTextResponse, summary="Analyse trade opportunities for an Indian sector")
async def analyze_sector(sector: str, request: Request, user_id: str = Depends(check_rate_limit)) -> PlainTextResponse:
    """Returns a Markdown report. Auth: Bearer token. Sectors: pharmaceuticals, technology, agriculture, textiles, automobiles, energy, finance, healthcare, manufacturing, chemicals, metals, real_estate, fmcg, infrastructure, it_services, telecom, retail, defence, aviation, gems_jewellery"""
    sector_clean = sector.lower().strip().replace(" ", "_").replace("-", "_")
    if sector_clean not in VALID_SECTORS:
        raise HTTPException(status_code=400, detail=f"Unknown sector '{sector}'. Valid sectors: {', '.join(sorted(VALID_SECTORS))}")
    session_id = get_or_create_session(request, user_id)
    now = time.time()
    if sector_clean in report_cache:
        entry = report_cache[sector_clean]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            return PlainTextResponse(content=entry["report"], headers={"X-Session-ID": session_id, "X-Cache": "HIT", "Content-Type": "text/markdown; charset=utf-8"})
    report = await analyse_with_groq(sector_clean)
    report_cache[sector_clean] = {"timestamp": now, "report": report}
    return PlainTextResponse(content=report, headers={"X-Session-ID": session_id, "X-Cache": "MISS", "Content-Type": "text/markdown; charset=utf-8"})

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "cached_sectors": list(report_cache.keys()), "active_sessions": len(session_store)}

@app.get("/sessions/me", summary="Your session info")
async def my_session(request: Request, user_id: str = Depends(get_current_user)):
    session_id = request.headers.get("X-Session-ID")
    session = session_store.get(session_id) if session_id else None
    remaining = RATE_LIMIT_REQUESTS - len([t for t in rate_limit_store[user_id] if t > time.time() - RATE_LIMIT_WINDOW_SECONDS])
    return {"user_id": user_id, "session": session, "rate_limit": {"limit": RATE_LIMIT_REQUESTS, "window_seconds": RATE_LIMIT_WINDOW_SECONDS, "remaining": max(remaining, 0)}}
