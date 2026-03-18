"""
Trade Opportunities API
FastAPI service that analyzes market data and provides trade opportunity
insights for specific sectors in India.
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

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("trade_api")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

def get_gemini_url() -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash-001:generateContent?key={GEMINI_API_KEY}"
    )

# Simple pre-shared API keys for demo auth
VALID_API_KEYS: dict[str, str] = {
    "demo-key-001": "demo_user_1",
    "demo-key-002": "demo_user_2",
    "guest-key-000": "guest",
}

# Rate-limit settings
RATE_LIMIT_REQUESTS: int = 5
RATE_LIMIT_WINDOW_SECONDS: int = 60

# Cache TTL
CACHE_TTL_SECONDS: int = 600

# ─────────────────────────────────────────
# In-memory stores
# ─────────────────────────────────────────
rate_limit_store: dict[str, list[float]] = defaultdict(list)
session_store: dict[str, dict] = {}
report_cache: dict[str, dict] = {}

# ─────────────────────────────────────────
# Valid sectors
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
        logger.warning("Unauthorised access attempt with token: %s", token[:8])
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return user_id


# ─────────────────────────────────────────
# Rate-limit dependency
# ─────────────────────────────────────────
def check_rate_limit(user_id: str = Depends(get_current_user)) -> str:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

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
# Gemini analysis
# ─────────────────────────────────────────
async def analyse_with_gemini(sector: str) -> str:
    """Send prompt to Gemini and get a markdown report."""

    prompt = f"""You are a senior trade analyst specialising in Indian markets.

Sector under analysis: **{sector.upper()}**
Today's date: {datetime.utcnow().strftime('%B %d, %Y')}

Produce a **structured Markdown trade-opportunity report** with exactly these sections:

# Trade Opportunity Report - {sector.title()} Sector (India)

## 1. Executive Summary
(3-4 sentences covering the most important insight)

## 2. Sector Overview
(Current state, key players, market size)

## 3. Current Market Trends
(Bullet list of 5-7 recent trends)

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
        gemini_url = get_gemini_url()
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                gemini_url,
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
        return data["candidates"][0]["content"]["parts"][0]["text"]

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

    **Auth**: Pass your API key as a Bearer token in the Authorization header.

    **Available sectors**: pharmaceuticals, technology, agriculture, textiles,
    automobiles, energy, finance, healthcare, manufacturing, chemicals, metals,
    real_estate, fmcg, infrastructure, it_services, telecom, retail, defence,
    aviation, gems_jewellery
    """
    # Input validation
    sector_clean = sector.lower().strip().replace(" ", "_").replace("-", "_")
    if sector_clean not in VALID_SECTORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown sector '{sector}'. "
                f"Valid sectors: {', '.join(sorted(VALID_SECTORS))}"
            ),
        )

    # Session tracking
    session_id = get_or_create_session(request, user_id)
    logger.info("Request | user=%s | session=%s | sector=%s", user_id, session_id, sector_clean)

    # Cache check
    now = time.time()
    if sector_clean in report_cache:
        entry = report_cache[sector_clean]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            logger.info("Cache hit for sector: %s", sector_clean)
            return PlainTextResponse(
                content=entry["report"],
                headers={
                    "X-Session-ID": session_id,
                    "X-Cache": "HIT",
                    "Content-Type": "text/markdown; charset=utf-8",
                },
            )

    # LLM analysis
    report = await analyse_with_gemini(sector_clean)

    # Cache store
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
