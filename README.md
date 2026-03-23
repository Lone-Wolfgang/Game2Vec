# Steam Recommendation Engine

A Streamlit app that recommends Steam games using Reciprocal Rank Fusion (RRF) over four retrieval signals: title similarity, description semantics, user tags, and review sentiment.

## Architecture

```
app/
├── app.py              # Entry point
├── config.py           # DB credentials, model names, CSS
├── resources.py        # Cached resource loaders (DB, models, retriever)
└── modules/
    ├── db.py           # PostgreSQL connection wrapper
    ├── rag.py          # Retriever (RRF + rankers) and Generator (RAG)
    ├── ux.py           # Streamlit UI — SessionManager
    └── pmi.py          # PMI tag-graph explorer component
```

## How it works

**Retrieval** (`rag.py`) — `Retriever.rrf()` fuses four independent rankers:

| Ranker | Signal | Method |
|---|---|---|
| Title | Trigram similarity on app name | PostgreSQL `pg_trgm` |
| Description | Semantic similarity on short description | ANN (pgvector) → cross-encoder rerank |
| Tags | PMI-weighted user tag matching | Co-occurrence graph built at startup |
| Reviews | Semantic similarity on user reviews | ANN (pgvector) → cross-encoder rerank |

Each ranker returns a ranked list; RRF combines them as `score = Σ 1/(k + rank)`. Individual rankers can be toggled on/off in the UI.

**Generation** (`rag.py`) — `Generator.rag()` prompts a small causal LM (LFM2-1.2B) with the game description and top review snippets to produce a personalised pitch.

**PMI tag graph** (`pmi.py`) — `UsertagRanker` builds a pointwise mutual information co-occurrence graph over all app–tag pairs at startup. The graph is exposed as an interactive D3 force-directed visualisation via `st.components.v2`. Clicking a node adds the tag to the active filter; the neighbour panel shows the strongest PMI-associated tags for the current selection.

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL (Docker recommended) with pgvector extension
- The database should contain: `applications`, `applications_usertags`, `usertags`, `reviews`, `sentences`

### Install

```bash
pip install -r requirements.txt
```

### Configure

Edit `config.py`:

```python
HOST     = "localhost"
PORT     = 5433
DB       = "steam_rec"
USER     = "postgres"
PASSWORD = "postgres"

BIENCODER = {
    "base":  "BAAI/bge-base-en-v1.5",   # query + description embeddings
    "small": "BAAI/bge-small-en-v1.5",  # sentence-level embeddings
}
RERANKER = "jinaai/jina-reranker-v1-turbo-en"
GENAI    = "LiquidAI/LFM2-1.2B"
```

### Run

```bash
streamlit run app.py
```

Models are downloaded from HuggingFace Hub on first run and cached by `@st.cache_resource`.

## Usage

1. **Search** — type a natural language query (e.g. *"cooperative survival crafting"*). The app suggests relevant user tags and runs a full RRF search automatically.
2. **User Tags** — manually add or remove Steam user tags to refine results.
3. **Tag Graph** — opens a PMI co-occurrence graph. Click any node or neighbour row to add that tag to the active filter; changes sync back to the User Tags multiselect in real time.
4. **Rankers** — the ⚙️ expander lets you toggle individual rankers on/off. The Search button re-appears whenever tags or ranker selection diverge from the last search.
5. **Results** — the top recommendation is shown with review snippets and an AI-generated pitch. Click *Generate* to produce the pitch; click a card in "You may also like" to pin it.

## Requirements

```
streamlit
pandas
psycopg2-binary
sentence-transformers
transformers
torch
accelerate
```