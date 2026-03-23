"""
config.py
=========
Central configuration: database credentials, model identifiers, and app-wide CSS.

Edit this file to point the app at a different database or swap in alternative models.
"""

HOST     = "localhost"
PORT     = 5433
DB       = "steam_rec"
USER     = "postgres"
PASSWORD = "postgres"

CONNECTION = dict(host=HOST, port=PORT, database=DB, user=USER, password=PASSWORD)

TOP_K = 5

# Generative model for RAG pitches
GENAI = "LiquidAI/LFM2-1.2B"

# Cross-encoder used to rerank ANN candidates for Description and Review rankers
RERANKER = "jinaai/jina-reranker-v1-turbo-en"

# Bi-encoders: base for query/description/review embeddings,
# small for sentence-level snippet retrieval
BIENCODER = {
    "base":  "BAAI/bge-base-en-v1.5",
    "small": "BAAI/bge-small-en-v1.5",
}

TITLE = "Steam Recommendation Engine"

# ---------------------------------------------------------------------------
# Global CSS injected by SessionManager._init_page()
# ---------------------------------------------------------------------------
CSS = """
<style>
section.main > div { max-width: 100% !important; padding-left: 1rem; padding-right: 1rem; }
.block-container { max-width: 1600px !important; padding-top: 2rem; padding-left: 2rem; padding-right: 2rem; }
section.main { align-items: flex-start !important; }
</style>

<style>
button[kind="secondary"] {
    background: none !important; border: none !important; padding: 0 !important;
    text-align: left !important; font-size: 1.6rem !important; font-weight: 600;
    color: white !important; cursor: pointer;
}
button[kind="secondary"]:hover { text-decoration: underline; }
</style>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: radial-gradient(circle at top left, #1f2a3a 0%, #171a21 60%);
    color: #c7d5e0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
h1 { font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: 0.5px; }
h2, h3 { color: #ffffff; font-weight: 600; }
p { color: #acb2b8; line-height: 1.5; font-size: 0.95rem; }
hr { border: none; border-top: 1px solid #2a475e; margin: 1.2rem 0; }

.game-card {
    background: linear-gradient(145deg, #16202d, #1b2838);
    padding: 1.2rem; border-radius: 10px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.game-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.45); }

img { border-radius: 6px; margin-top: 0.5rem; margin-bottom: 0.75rem; }

.pill {
    display: inline-block; padding: 0.25em 0.75em; margin: 0.2em;
    font-size: 0.8rem; font-weight: 500; border-radius: 999px;
    background-color: #2a475e; color: #66c0f4; border: 1px solid #3b5a72;
    white-space: nowrap; transition: background-color 0.15s ease;
}
.pill:hover { background-color: #3b5a72; }
.tags-container { text-align: left; margin-top: 0.5em; margin-bottom: 0.75em; }

section[data-testid="stSidebar"] {
    background: #111822; border-right: 1px solid #2a475e;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff; }
section[data-testid="stSidebar"] button {
    background-color: #2a475e !important; border: 1px solid #3b5a72 !important;
    color: #c7d5e0 !important; border-radius: 6px !important;
}
section[data-testid="stSidebar"] button:hover { background-color: #3b5a72 !important; }

button:focus { outline: none !important; box-shadow: none !important; }
</style>
"""