import sys
import multiprocessing as mp
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
import psycopg2
import psycopg2.extras
from tqdm.auto import tqdm
import spacy
from sentence_transformers import SentenceTransformer

from project.app.modules.index import (
    Preprocessor,
    SentenceExtractor,
)

# ====================================================
# CONFIG
# ====================================================

DB = dict(
    dbname="steam_rec",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433,
)

SCHEMA = "public"
TABLE = "reviews"

REVIEWS = Path(r'/Users/jwkle/Documents/Steam/project/data/scraped/assembled/reviews.parquet')

RESTART = False          # True  → drop table + index and start from scratch
                         # False → resume from where we left off

STREAM_BATCH_SIZE = 64
BUILD_INDEX = True
INDEX_RAM = "8GB"
PARALLEL_WORKERS = 4

MODEL_NAME = "BAAI/bge-base-en-v1.5"
NORMALIZE_EMBEDDINGS = True

# ====================================================
# TABLE SETUP
# ====================================================

def reset_table(conn):
    """Drop table and index if RESTART=True."""
    index_name = f"{TABLE}_embedding_hnsw"
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS public.{index_name};")
        cur.execute(f"DROP TABLE IF EXISTS public.{TABLE};")
        conn.commit()
    print(f"⚠ Dropped {TABLE} table and index")


def setup_table(conn, embed_dim: int):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS public.{TABLE} (
                review_id        TEXT PRIMARY KEY,
                app_id           INTEGER NOT NULL,
                review           TEXT,
                review_embedding VECTOR({embed_dim}),
                FOREIGN KEY (app_id) REFERENCES public.applications(app_id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    print(f"✓ {TABLE} table ready")


def build_hnsw_index(conn):
    index_name = f"{TABLE}_embedding_hnsw"
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename = '{TABLE}'
            AND indexname = '{index_name}'
        """)
        if cur.fetchone():
            print("✓ HNSW index already exists")
            return

    print("Analyzing table...")
    with conn.cursor() as cur:
        cur.execute(f"ANALYZE public.{TABLE};")
        conn.commit()

    print("Building HNSW index...")
    with conn.cursor() as cur:
        cur.execute(f"SET maintenance_work_mem = '{INDEX_RAM}';")
        cur.execute(f"SET max_parallel_maintenance_workers = {PARALLEL_WORKERS};")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON public.{TABLE}
            USING hnsw (review_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """)
        conn.commit()
    print("✓ HNSW index created")

# ====================================================
# RESUME SUPPORT
# ====================================================

def get_completed_ids(conn) -> set:
    """Return the set of review_ids already in the DB."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT review_id FROM public.{TABLE};")
        return {row[0] for row in cur.fetchall()}

# ====================================================
# LOAD REVIEWS FROM PARQUET
# ====================================================

def load_reviews(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["review_id", "app_id", "review"])
    df["review_id"] = df["review_id"].astype(str)
    df["app_id"] = df["app_id"].astype(int)
    df = df.dropna(subset=["review"])
    df = df[df["review"].str.strip() != ""]
    return df.reset_index(drop=True)

# ====================================================
# INSERT
# ====================================================

def insert_batch(conn, df: pd.DataFrame):
    values = [
        (
            row["review_id"],
            int(row["app_id"]),
            row["review"],
            row["review_embedding"],
        )
        for _, row in df.iterrows()
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO public.{TABLE} (review_id, app_id, review, review_embedding)
            VALUES %s
            ON CONFLICT (review_id) DO NOTHING
            """,
            values,
            template="(%s, %s, %s, %s::vector)",
        )
    conn.commit()

# ====================================================
# MAIN
# ====================================================

def main():

    conn = psycopg2.connect(**DB)

    # ── Load models ──────────────────────────────────
    print("Loading NLP + embedding models...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    nlp.enable_pipe("senter")

    sentence_extractor = SentenceExtractor(nlp, batch_size=2000, n_process=1)

    review_model = SentenceTransformer(MODEL_NAME)
    embed_dim = review_model.get_sentence_embedding_dimension()
    print(f"✓ Embedding dimension: {embed_dim}")

    preprocessor = Preprocessor(
        sentence_extractor=sentence_extractor,
        model=review_model,
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS, "show_progress_bar": False},
    )

    # ── Restart or resume ────────────────────────────
    if RESTART:
        reset_table(conn)

    setup_table(conn, embed_dim)

    # ── Load parquet ─────────────────────────────────
    print(f"Loading reviews from {REVIEWS} ...")
    reviews_df = load_reviews(REVIEWS)
    print(f"✓ {len(reviews_df):,} reviews loaded from parquet")

    # ── Filter already-completed reviews ─────────────
    if not RESTART:
        completed = get_completed_ids(conn)
        if completed:
            reviews_df = reviews_df[~reviews_df["review_id"].isin(completed)].reset_index(drop=True)
            print(f"✓ Resuming — skipping {len(completed):,} already inserted, {len(reviews_df):,} remaining")
        else:
            print("✓ No existing rows found — starting fresh")

    if reviews_df.empty:
        print("✓ Nothing to do — all reviews already inserted")
        if BUILD_INDEX:
            build_hnsw_index(conn)
        conn.close()
        return

    # ── Process + insert in batches ──────────────────
    total = len(reviews_df)
    total_inserted = 0

    with tqdm(total=total, desc="Embedding reviews", unit="reviews") as pbar:
        for start in range(0, total, STREAM_BATCH_SIZE):
            batch = reviews_df.iloc[start:start + STREAM_BATCH_SIZE].copy()

            processed = preprocessor.process_reviews(batch)

            if processed.empty:
                pbar.update(len(batch))
                continue

            insert_batch(conn, processed)
            total_inserted += len(processed)
            pbar.update(len(batch))

    print(f"✓ {total_inserted:,} reviews embedded and inserted into {TABLE}")

    # ── Build index ──────────────────────────────────
    if BUILD_INDEX:
        build_hnsw_index(conn)

    conn.close()
    print("✅ Pipeline complete")


# ====================================================
if __name__ == "__main__":
    mp.freeze_support()
    main()