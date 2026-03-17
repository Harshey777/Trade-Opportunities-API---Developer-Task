"""
Trade Opportunities API
FastAPI service that analyzes market data and provides trade opportunity
insights for specific sectors in India.
"""

import time
import uuid
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("trade_api")

# ─────────────────────────────────────────
# Config (swap in real keys via env vars)
# ─────────────────────────────────────────
import os

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL:   str = "gemini-1.5-flash"
GEMINI_URL:     str = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

# Simple pre-shared API keys for demo auth
# In production these would live in a secrets manager
VALID_API_KEYS: dict[str, str] = {
    "demo-key-001": "demo_user_1",
    "demo-key-002": "demo_user_2",
    "guest-key-000": "guest",
}

# Rate-limit: max requests per window
RATE_LIMIT_REQUESTS: int = 5
RATE_LIMIT_WINDOW_SECONDS: int = 60

# ─────────────────────────────────────────
# In-memory stores
# ─────────────────────────────────────────
# { user_id: [(timestamp, …), …] }
rate_limit_store: dict[str, list[float]] = defaultdict(list)

# { session_id: {created, user_id, requests: int} }
session_store: dict[str, dict] = {}

# { sector: {timestamp, report} }  – simple cache (TTL: 10 min)
report_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS: int = 600

# ─────────────────────────────────────────
# Valid sectors (input validation)
# ─────────────────────────────────────────
VALID_SECTORS: set[str] = {
    "pharmaceuticals", "technology", "agriculture", "textiles",
    "automobiles", "energy", "finance", "healthcare", "manufacturing",
    "chemicals", "metals", "real_estate", "fmcg", "infrastructure",
    "it_services", "telecom", "retail", "defence", "aviation", "gems_jewellery",
}

# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────
app = FastAPI(
    title="Trade Opportunities API",
    description=(
        "Analyzes market data and provides structured trade opportunity "
        "insights for specific sectors in India."
    ),
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

# ─────────────────────────────────────────
# Auth dependency
# ─────────────────────────────────────────
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials
    user_id = VALID_API_KEYS.get(token)
    if not user_id:
        logger.warning("Unauthorised access attempt with token: %s…", token[:8])
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return user_id


# ─────────────────────────────────────────
# Rate-limit dependency
# ─────────────────────────────────────────
def check_rate_limit(user_id: str = Depends(get_current_user)) -> str:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    # Purge old entries
    rate_limit_store[user_id] = [
        t for t in rate_limit_store[user_id] if t > window_start
    ]

    if len(rate_limit_store[user_id]) >= RATE_LIMIT_REQUESTS:
        oldest = rate_limit_store[user_id][0]
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - oldest)) + 1
        logger.warning("Rate limit hit for user: %s", user_id)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS}s. "
                f"Retry after {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    rate_limit_store[user_id].append(now)
    return user_id


# ─────────────────────────────────────────
# Session helper
# ─────────────────────────────────────────
def get_or_create_session(request: Request, user_id: str) -> str:
    session_id = request.headers.get("X-Session-ID")
    if session_id and session_id in session_store:
        session_store[session_id]["requests"] += 1
        session_store[session_id]["last_seen"] = datetime.utcnow().isoformat()
    else:
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "requests": 1,
        }
    return session_id


# ─────────────────────────────────────────
# DuckDuckGo search (no API key needed)
# ─────────────────────────────────────────
async def search_web(query: str, max_results: int = 8) -> list[dict]:
    """Fetch DuckDuckGo Instant Answer + text snippets."""
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                headers={"User-Agent": "TradeOpportunitiesAPI/1.0"},
            )
            data = resp.json()

        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({"title": data.get("Heading", query), "snippet": abstract})

        for topic in data.get("RelatedTopics", [])[:max_results]:
            text = topic.get("Text", "")
            if text:
                results.append({"title": topic.get("FirstURL", ""), "snippet": text})

    except Exception as exc:
        logger.error("Web search error: %s", exc)

    return results


# ─────────────────────────────────────────
# Gemini analysis
# ─────────────────────────────────────────
async def analyse_with_gemini(sector: str, search_results: list[dict]) -> str:
    """Send collected snippets to Gemini and get a markdown report."""
    snippets = "\n".join(
        f"- {r['snippet']}" for r in search_results if r.get("snippet")
    ) or "No live data available; use general knowledge."

    prompt = f"""You are a senior trade analyst specialising in Indian markets.

Sector under analysis: **{sector.upper()}**
Today's date: {datetime.utcnow().strftime('%B %d, %Y')}

Web data collected:
{snippets}

Produce a **structured Markdown trade-opportunity report** with exactly these sections:

# Trade Opportunity Report – {sector.title()} Sector (India)

## 1. Executive Summary
(3–4 sentences covering the most important insight)

## 2. Sector Overview
(Current state, key players, market size)

## 3. Current Market Trends
(Bullet list of 5–7 recent trends)

## 4. Trade Opportunities
(Table with columns: Opportunity | Target Markets | Estimated Value | Confidence)

## 5. Export Opportunities
(Top export prospects with brief rationale)

## 6. Import Opportunities
(Key imports that present arbitrage or strategic value)

## 7. Risks & Challenges
(Bullet list)

## 8. Policy & Regulatory Landscape
(Relevant GOI policies, PLI schemes, tariffs, bilateral agreements)

## 9. Recommended Actions
(Numbered list of concrete next steps for a trader/investor)

## 10. Data Sources & Disclaimer
(Mention data freshness, caveats)

Use clear, concise, professional language. Be specific to India.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 4096,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GEMINI_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code != 200:
            logger.error("Gemini API error %s: %s", resp.status_code, resp.text[:300])
            raise HTTPException(
                status_code=502,
                detail=f"Gemini API returned {resp.status_code}. Check your API key.",
            )

        data = resp.json()
        return (
            data["candidates"][0]["content"]["parts"][0]["text"]
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get(
    "/analyze/{sector}",
    response_class=PlainTextResponse,
    summary="Analyse trade opportunities for an Indian sector",
    responses={
        200: {"description": "Markdown trade-opportunity report"},
        400: {"description": "Invalid sector name"},
        401: {"description": "Unauthorised"},
        429: {"description": "Rate limit exceeded"},
        502: {"description": "Upstream API error"},
    },
)
async def analyze_sector(
    sector: str,
    request: Request,
    user_id: str = Depends(check_rate_limit),
) -> PlainTextResponse:
    """
    Returns a structured Markdown report with current trade opportunities
    for the requested sector in India.

    **Auth**: Pass your API key as a Bearer token in the `Authorization` header.

    **Available sectors**: pharmaceuticals, technology, agriculture, textiles,
    automobiles, energy, finance, healthcare, manufacturing, chemicals, metals,
    real_estate, fmcg, infrastructure, it_services, telecom, retail, defence,
    aviation, gems_jewellery
    """
    # ── Input validation ──────────────────
    sector_clean = sector.lower().strip().replace(" ", "_").replace("-", "_")
    if sector_clean not in VALID_SECTORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown sector '{sector}'. "
                f"Valid sectors: {', '.join(sorted(VALID_SECTORS))}"
            ),
        )

    # ── Session tracking ──────────────────
    session_id = get_or_create_session(request, user_id)
    logger.info("Request | user=%s | session=%s | sector=%s", user_id, session_id, sector_clean)

    # ── Cache check ───────────────────────
    now = time.time()
    if sector_clean in report_cache:
        entry = report_cache[sector_clean]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            logger.info("Cache hit for sector: %s", sector_clean)
            cached_report = entry["report"]
            return PlainTextResponse(
                content=cached_report,
                headers={
                    "X-Session-ID": session_id,
                    "X-Cache": "HIT",
                    "Content-Type": "text/markdown; charset=utf-8",
                },
            )

    # ── Data collection ───────────────────
    search_queries = [
        f"India {sector_clean} sector trade opportunities 2024 2025",
        f"India {sector_clean} exports imports market analysis",
        f"India {sector_clean} industry growth trends recent news",
    ]

    all_results: list[dict] = []
    for query in search_queries:
        results = await search_web(query)
        all_results.extend(results)

    logger.info("Collected %d snippets for sector: %s", len(all_results), sector_clean)

    # ── LLM analysis ─────────────────────
    report = await analyse_with_gemini(sector_clean, all_results)

    # ── Cache store ───────────────────────
    report_cache[sector_clean] = {"timestamp": now, "report": report}

    return PlainTextResponse(
        content=report,
        headers={
            "X-Session-ID": session_id,
            "X-Cache": "MISS",
            "Content-Type": "text/markdown; charset=utf-8",
        },
    )


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "cached_sectors": list(report_cache.keys()),
        "active_sessions": len(session_store),
    }


@app.get("/sessions/me", summary="Your session info")
async def my_session(
    request: Request,
    user_id: str = Depends(get_current_user),
):
    session_id = request.headers.get("X-Session-ID")
    session = session_store.get(session_id) if session_id else None
    remaining = RATE_LIMIT_REQUESTS - len(
        [t for t in rate_limit_store[user_id]
         if t > time.time() - RATE_LIMIT_WINDOW_SECONDS]
    )
    return {
        "user_id": user_id,
        "session": session,
        "rate_limit": {
            "limit": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "remaining": max(remaining, 0),
        },
    }
