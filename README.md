# Buddy

AI-powered Stock Market Analyst for Indian equities.

Buddy answers three types of questions:

1. **Single stock** — *"How is Reliance doing?"*
2. **Portfolio** — *"How is my portfolio doing?"*
3. **Comparison** — *"Compare Tata Motors and Mahindra. Which has the stronger setup?"*

## Architecture

```
User → Streamlit → FastAPI → Query Parser → LangGraph Orchestrator
                                              ↓ (parallel)
                         Fundamental │ Technical │ Sentiment Analysts
                                              ↓
                                    Master Analyst → Decision Engine → Response
```

### Design principles

| Layer | Responsibility |
|---|---|
| **Data layer** | Fetch quotes, OHLCV, financials, news — swappable providers |
| **Analysis layer** | Deterministic Python calculations (indicators, metrics, scoring) |
| **Agents** | LLM interpretation of structured data — never raw number crunching |
| **Graph** | LangGraph orchestration with parallel analyst execution |
| **API** | FastAPI — usable independently of Streamlit |
| **Frontend** | Streamlit MVP UI |

### Project structure

```
Buddy/
├── app/
│   ├── api/              # FastAPI routes, middleware, request schemas
│   ├── agents/           # Analyst agents + query parser + LLM client
│   ├── graph/            # LangGraph state, nodes, workflow
│   ├── data/             # Yahoo Finance, web search, SQLite cache
│   ├── analysis/         # Indicators, metrics, scoring (deterministic)
│   ├── models/           # Pydantic domain schemas
│   ├── config/           # Settings and logging configuration
│   ├── services/         # Dependency injection container
│   └── main.py           # FastAPI app factory
├── frontend/
│   ├── streamlit_app.py  # Streamlit UI
│   └── ui_helpers.py     # Formatting helpers for the UI
├── tests/                # Pytest suite (mock data, no live LLM in CI)
├── docs/
│   └── API.md            # Endpoint reference and examples
├── main.py               # Uvicorn entry point
├── pytest.ini
├── requirements.txt
└── .env.example
```

## Tech stack

- **Python** · **LangGraph** · **FastAPI** · **Streamlit**
- **Yahoo Finance** (market data) · **DuckDuckGo** (news search)
- **Groq** LLM (configurable via env)
- **SQLite** cache (MVP) · **Pydantic** models

## Getting started

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Set GROQ_API_KEY in .env for live LLM interpretation

# 4. Run API
python main.py
# → http://localhost:8000/api/health
# → http://localhost:8000/docs

# 5. Run Streamlit (separate terminal)
streamlit run frontend/streamlit_app.py
```

> **Note:** `http://localhost:8000/api` has no route. Use `/api/health` or `/docs`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/analyze` | Analyze a stock from NL query |
| `POST` | `/api/compare` | Compare multiple stocks |
| `POST` | `/api/portfolio` | Analyze portfolio holdings |

See [docs/API.md](docs/API.md) for request/response examples.

### Quick examples

```bash
curl http://localhost:8000/api/health

curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"How is Reliance doing?\"}"

curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d "{\"stocks\": [\"TATAMOTORS.NS\", \"M&M.NS\"]}"

curl -X POST http://localhost:8000/api/portfolio \
  -H "Content-Type: application/json" \
  -d "{\"holdings\": [{\"symbol\": \"RELIANCE.NS\", \"quantity\": 10, \"buy_price\": 1000}]}"
```

## Testing

Tests use **mock market data**, **mock news search**, and **MockLLMClient** — no live Groq or Yahoo Finance calls in the default suite.

```bash
pytest                    # run all tests
pytest -v                 # verbose
pytest tests/test_graph_workflow.py  # single module
```

Configuration lives in `pytest.ini` (`asyncio_mode = auto`).

## Logging

Logging is configured at app startup via `app/config/logging_config.py`.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, …) |
| `LOG_FILE` | unset | Optional path for file logging (e.g. `logs/buddy.log`) |

HTTP requests are logged by `RequestLoggingMiddleware` (method, path, status, duration). Analyze/compare/portfolio routes log request context and outcomes.

Example:

```env
LOG_LEVEL=DEBUG
LOG_FILE=logs/buddy.log
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERR_CONNECTION_REFUSED` on port 8000 | API not running | Run `python main.py` in a separate terminal |
| 404 on `/api` | No route at prefix root | Use `/api/health` or `/docs` |
| Streamlit "Cannot reach API" | Backend down or wrong URL | Check sidebar API URL; default is `http://localhost:8000/api` |
| Mock LLM responses in production | `GROQ_API_KEY` not set | Set key in `.env`; without it Buddy uses `MockLLMClient` |
| Slow first request | Cold cache + live data fetch | Normal; subsequent calls use SQLite cache |

## Build phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Project structure + config + health endpoint | ✅ Architecture |
| 2 | Yahoo Finance + caching | ✅ |
| 3 | Technical indicator engine | ✅ |
| 4 | Technical Analyst | ✅ |
| 5 | Fundamental data + Analyst | ✅ |
| 6 | Web search + Sentiment Analyst | ✅ |
| 7 | LangGraph orchestrator + parallel execution | ✅ |
| 8 | Master Analyst + Decision Engine | ✅ |
| 9 | Portfolio Analyzer | ✅ |
| 10 | Comparison workflow | ✅ |
| 11 | Streamlit UI | ✅ |
| 12 | Testing, logging, docs | ✅ |

## Scoring

Default weights (configurable in `.env`):

- Fundamental: **40%**
- Technical: **35%**
- Sentiment: **15%**
- Risk: **10%**

Ratings: Strong Buy · Buy · Hold · Avoid

> Analytical outputs only — not guaranteed predictions.

## License

Private — MVP development.
