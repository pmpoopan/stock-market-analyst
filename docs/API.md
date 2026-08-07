# Buddy API Reference

Base URL: `http://localhost:8000/api` (configurable via `API_HOST`, `API_PORT`, `API_PREFIX`).

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Health

### `GET /health`

Liveness probe — does not call external services.

**Response**

```json
{
  "status": "ok",
  "app": "Buddy",
  "version": "0.1.0"
}
```

## Analyze stock

### `POST /analyze`

Natural-language query for a single Indian equity.

**Request**

```json
{
  "query": "How is Reliance doing?"
}
```

**Response** (`StockAnalysisResponse`)

- `symbol`, `name`, `current_price`
- `decision` — overall score, rating, key reasons, major risks
- `fundamental`, `technical`, `sentiment`, `master` — full analyst outputs
- `sources` — news URLs used in sentiment analysis

**Errors**

- `400` — query could not be parsed or analysis failed
- `422` — missing or invalid request body

## Compare stocks

### `POST /compare`

Side-by-side comparison of two or more symbols.

**Request**

```json
{
  "stocks": ["TATAMOTORS.NS", "M&M.NS"]
}
```

Symbols should use Yahoo Finance format (e.g. `RELIANCE.NS` for NSE).

**Response** (`StockComparisonResult`)

- Score maps: `fundamental_scores`, `technical_scores`, `sentiment_scores`, `overall_scores`
- Narratives: `valuation_comparison`, `growth_comparison`, `risk_comparison`, `technical_trend_comparison`
- `winner` — leading symbol if scores differ by ≥2 points; otherwise `null`
- `relative_assessment` — summary text

**Errors**

- `400` — comparison could not be completed
- `422` — fewer than two symbols

## Analyze portfolio

### `POST /portfolio`

Aggregate analysis for a list of holdings.

**Request**

```json
{
  "holdings": [
    { "symbol": "RELIANCE.NS", "quantity": 10, "buy_price": 1000 },
    { "symbol": "INFY.NS", "quantity": 50, "buy_price": 1500 }
  ]
}
```

**Response** (`PortfolioAnalysisResult`)

- Per-holding P&L, allocation, and decision
- `total_invested`, `total_current_value`, `total_pnl`, `total_pnl_percent`
- `portfolio_score`, `strongest_holdings`, `weakest_holdings`
- `sector_concentration`, `portfolio_risk`, `summary`

**Errors**

- `400` — portfolio analysis failed
- `422` — empty holdings list

## LangGraph pipeline

All analysis endpoints run the same orchestration graph:

1. Parse query / symbols / holdings
2. Fetch quotes
3. Run fundamental, technical, and sentiment analysts **in parallel**
4. Master analyst synthesis
5. Decision engine scoring
6. Comparison or portfolio aggregation (when applicable)

Typical latency: several seconds per request when using live data and LLM interpretation.

## Environment

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Live LLM via Groq; omit to use mock LLM |
| `API_BASE_URL` | Streamlit frontend backend URL |
| `LOG_LEVEL` / `LOG_FILE` | Server logging |
| `CACHE_ENABLED` | SQLite cache for market data and search |

See `.env.example` for the full list.
