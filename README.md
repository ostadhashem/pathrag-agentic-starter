# PathRAG Agentic Starter

A GitHub-ready, minimal starter for building an **agentic architecture** around **Path-RAG** using two frameworks:

- **LangGraph (LangChain)** — production-grade, stateful graph orchestration.
- **AutoGen** — fast multi-agent prototyping (planner/critic/executor).

This repo ships with:
- A clean Python package layout (`src/`).
- Mock **tools** that simulate HistoCartography, LLaVA-Med, and a text **reasoner**.
- Two runnable templates:
  - `scripts/run_langgraph.py`
  - `scripts/run_autogen.py`

> Swap the mocked tool functions with your real implementations (REST calls or Python libs).

---

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your API key (if you plan to call OpenAI for the reasoner)
cp src/pathrag/workflows/config/.env.example .env
# then edit .env and add OPENAI_API_KEY=...

# 4) Run either template
python scripts/run_langgraph.py
python scripts/run_autogen.py
```

### Expected output
Both scripts use mocked tools and should print a final answer like:
```
FINAL ANSWER: [mocked] keratinization (example)
```

---

## Repo structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── pathrag/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── langgraph_app.py     # LangGraph template
│       │   ├── autogen_app.py       # AutoGen template
│       │   └── tools.py             # Mocked tools (swap with real ones)
│       ├── workflows/
│       │   ├── config/
│       │   │   └── .env.example
│       │   └── examples/
│       └── utils/
│           ├── __init__.py
│           └── logging.py
├── scripts/
│   ├── run_langgraph.py
│   └── run_autogen.py
└── tests/
    └── test_smoke.py
```

---

## How to replace mocks with real components

- **HistoCartography**
  - Replace `tools.run_histocartography` with your pipeline or an HTTP client to your service.
  - Return a dict of the form: `{"is_he": bool, "patches": ["<img>::patchA", "<img>::patchB", "<img>::patchC"]}`.

- **LLaVA-Med**
  - Replace `tools.run_llava_med_on_image/patch` with your local inference call or REST endpoint.

- **Reasoner (GPT-4-class)**
  - Keep `tools.gpt_reason` if you have an OpenAI key, or wire a different client (Anthropic, local vLLM).
  - For air‑gapped deployments, wrap your local LLM in an HTTP service and call it here.

---

## Notes
- This starter avoids heavy dependencies and keeps the orchestrators **CPU‑light**.
- For production, consider:
  - **Queues** (Redis/Rabbit) between orchestrator and GPU workers.
  - **Tracing** (LangSmith or OpenTelemetry).
  - **HITL** pause nodes in LangGraph or approvals via an outer workflow (e.g., n8n).