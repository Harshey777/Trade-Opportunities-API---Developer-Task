# Trade Opportunities API

A **FastAPI** service that analyses market data and returns structured Markdown trade-opportunity reports for specific sectors in India.

---

## Features

| Feature | Detail |
|---|---|
| Single endpoint | `GET /analyze/{sector}` |
| LLM | Google Gemini 1.5 Flash |
| Web search | DuckDuckGo (no key needed) |
| Auth | Bearer API-key |
| Rate limiting | 5 req / 60 s per user (in-memory) |
| Session tracking | UUID sessions via `X-Session-ID` header |
| Caching | 10-minute in-memory cache per sector |
| Storage | In-memory only |

---

## Quick Start

### 1 – Prerequisites

- Python 3.11+
- A free [Google AI Studio](https://aistudio.google.com/) account → get a **Gemini API key**

### 2 – Install

```bash
git clone <repo>
cd trade_api
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 – Configure

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

Or create a `.env` file and load it manually before starting.

### 4 – Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

---

## Authentication

Pass one of the demo API keys as a **Bearer token**:

| API Key | User |
|---|---|
| `demo-key-001` | demo_user_1 |
| `demo-key-002` | demo_user_2 |
| `guest-key-000` | guest |

```bash
curl -H "Authorization: Bearer demo-key-001" \
     http://localhost:8000/analyze/pharmaceuticals
```

> **Production tip:** Replace `VALID_API_KEYS` in `main.py` with a secrets-manager lookup or JWT verification.

---

## API Reference

### `GET /analyze/{sector}`

Returns a structured Markdown market-analysis report.

**Path parameter**

| Param | Type | Example |
|---|---|---|
| `sector` | string | `pharmaceuticals` |

**Valid sectors**

`agriculture` · `automobiles` · `aviation` · `chemicals` · `defence` · `energy` · `finance` · `fmcg` · `gems_jewellery` · `healthcare` · `infrastructure` · `it_services` · `manufacturing` · `metals` · `pharmaceuticals` · `real_estate` · `retail` · `technology` · `telecom` · `textiles`

**Headers**

| Header | Required | Description |
|---|---|---|
| `Authorization` | ✅ | `Bearer <api-key>` |
| `X-Session-ID` | Optional | Resume an existing session |

**Example request**

```bash
curl -s \
  -H "Authorization: Bearer demo-key-001" \
  http://localhost:8000/analyze/technology \
  -o technology_report.md
```

**Example response (truncated)**

```markdown
# Trade Opportunity Report – Technology Sector (India)

## 1. Executive Summary
India's technology sector continues to be a global powerhouse...

## 2. Sector Overview
...
```

**Response headers**

| Header | Value |
|---|---|
| `X-Session-ID` | UUID of your session |
| `X-Cache` | `HIT` or `MISS` |
| `Content-Type` | `text/markdown; charset=utf-8` |

---

### `GET /health`

Health check (no auth required).

```bash
curl http://localhost:8000/health
```

---

### `GET /sessions/me`

Returns your session info and remaining rate-limit quota.

```bash
curl -H "Authorization: Bearer demo-key-001" \
     http://localhost:8000/sessions/me
```

---

## Rate Limiting

- **5 requests per 60 seconds** per API key.
- On breach: `HTTP 429` with `Retry-After` header.

---

## Project Structure

```
trade_api/
├── main.py          # Full FastAPI application
├── requirements.txt
└── README.md
```

---

## Architecture

```
Client
  │
  ▼
FastAPI (main.py)
  ├── Auth middleware  (Bearer key validation)
  ├── Rate limiter     (in-memory sliding window)
  ├── Session tracker  (in-memory dict)
  │
  ├── GET /analyze/{sector}
  │     ├── Input validation  (allowlist of sectors)
  │     ├── Cache lookup      (10-min TTL)
  │     ├── DuckDuckGo search (3 queries → snippets)
  │     └── Gemini 1.5 Flash  (structured Markdown report)
  │
  ├── GET /health
  └── GET /sessions/me
```

---

## Error Codes

| Code | Meaning |
|---|---|
| 400 | Invalid / unknown sector name |
| 401 | Missing or invalid API key |
| 429 | Rate limit exceeded |
| 502 | Gemini API or web search failure |

---

## Customisation

- **Add sectors**: extend the `VALID_SECTORS` set in `main.py`.
- **Change rate limits**: update `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`.
- **Swap LLM**: replace `analyse_with_gemini()` with any OpenAI-compatible call.
- **Persist cache**: swap `report_cache` dict for Redis with `aioredis`.
