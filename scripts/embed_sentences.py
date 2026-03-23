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
TABLE = "sentences"

REVIEWS = Path(r'/Users/jwkle/Documents/Steam/project/data/scraped/assembled/reviews.parquet')
SENTENCES_CHECKPOINT = REVIEWS.parent / "sentences.parquet"

RESTART = False          # True  → drop table + index and start from scratch
                         # False → resume from where we left off

BATCH_SIZE = 128
BUILD_INDEX = True
INDEX_RAM = "12GB"
PARALLEL_WORKERS = 4

MODEL_NAME = "BAAI/bge-small-en-v1.5"
NORMALIZE_EMBEDDINGS = True

# ====================================================
# TABLE SETUP
# ====================================================

def reset_table(conn):
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
                sentence_id        TEXT PRIMARY KEY,
                review_id          TEXT NOT NULL,
                app_id             INTEGER NOT NULL,
                sentence           TEXT,
                sentence_embedding VECTOR({embed_dim}),
                FOREIGN KEY (review_id) REFERENCES public.reviews(review_id) ON DELETE CASCADE,
                FOREIGN KEY (app_id)    REFERENCES public.applications(app_id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    print(f"✓ {TABLE} table ready")


def build_hnsw_index(conn):
    index_name = f"{TABLE}_embedding_hnsw"

    with conn.cursor() as cur:
        # Acquire exclusive lock — only one process can proceed
        cur.execute("SELECT pg_try_advisory_lock(12345);")
        if not cur.fetchone()[0]:
            print("⚠ Another process is already building the index, skipping.")
            return

        # Check again inside the lock
        cur.execute(f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename = '{TABLE}'
            AND indexname = '{index_name}'
        """)
        if cur.fetchone():
            print("✓ HNSW index already exists")
            return

        print("Building HNSW index...")
        cur.execute(f"SET maintenance_work_mem = '{INDEX_RAM}';")
        cur.execute(f"SET max_parallel_maintenance_workers = {PARALLEL_WORKERS};")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON public.{TABLE}
            USING hnsw (sentence_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """)
        conn.commit()
    print("✓ HNSW index created")

# ====================================================
# RESUME SUPPORT
# ====================================================

def get_completed_sentence_ids(conn) -> set:
    with conn.cursor() as cur:
        cur.execute(f"SELECT sentence_id FROM public.{TABLE};")
        return {row[0] for row in cur.fetchall()}

# ====================================================
# INSERT
# ====================================================

def insert_batch(conn, df: pd.DataFrame):
    values = [
        (
            row["sentence_id"],
            row["review_id"],
            int(row["app_id"]),
            row["sentence"],
            row["sentence_embedding"],
        )
        for _, row in df.iterrows()
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO public.{TABLE} (sentence_id, review_id, app_id, sentence, sentence_embedding)
            VALUES %s
            ON CONFLICT (sentence_id) DO NOTHING
            """,
            values,
            template="(%s, %s, %s, %s, %s::vector)",
        )
    conn.commit()

# ====================================================
# PHASE 1 — SEGMENTATION
# ====================================================

def run_segmentation(preprocessor: Preprocessor) -> pd.DataFrame:
    """
    Segment all reviews into sentences and save checkpoint parquet.
    Skipped if checkpoint already exists.
    """
    if SENTENCES_CHECKPOINT.exists():
        print(f"✓ Checkpoint found — loading sentences from {SENTENCES_CHECKPOINT}")
        return pd.read_parquet(SENTENCES_CHECKPOINT)

    print(f"Loading reviews from {REVIEWS} ...")
    reviews_df = pd.read_parquet(REVIEWS, columns=["review_id", "app_id", "review"])
    reviews_df["review_id"] = reviews_df["review_id"].astype(str)
    reviews_df["app_id"] = reviews_df["app_id"].astype(int)
    reviews_df = reviews_df.dropna(subset=["review"])
    reviews_df = reviews_df[reviews_df["review"].str.strip() != ""].reset_index(drop=True)
    print(f"✓ {len(reviews_df):,} reviews loaded")

    print("Phase 1 — Segmenting all reviews...")
    sentences_df = preprocessor.extract_sentences(reviews_df)

    print(f"✓ {len(sentences_df):,} sentences extracted — saving checkpoint to {SENTENCES_CHECKPOINT}")
    sentences_df.to_parquet(SENTENCES_CHECKPOINT, index=False)

    return sentences_df

# ====================================================
# PHASE 2 — EMBED + INSERT
# ====================================================

def run_embedding(conn, sentences_df: pd.DataFrame, preprocessor: Preprocessor):
    """
    Embed sentences in batches of BATCH_SIZE and insert into DB.
    Skips already-inserted sentence_ids.
    """
    if not RESTART:
        completed = get_completed_sentence_ids(conn)
        if completed:
            sentences_df = sentences_df[~sentences_df["sentence_id"].isin(completed)].reset_index(drop=True)
            print(f"✓ Resuming — skipping {len(completed):,} already inserted, {len(sentences_df):,} remaining")
        else:
            print("✓ No existing rows found — starting fresh")

    if sentences_df.empty:
        print("✓ Nothing to do — all sentences already inserted")
        return

    total = len(sentences_df)
    total_inserted = 0

    print("Phase 2 — Embedding and inserting sentences...")
    with tqdm(total=total, desc="Embedding sentences", unit="sentences") as pbar:
        for start in range(0, total, BATCH_SIZE):
            batch = sentences_df.iloc[start:start + BATCH_SIZE].copy()

            embedded = preprocessor.embed_sentences(batch)
            insert_batch(conn, embedded)

            total_inserted += len(embedded)
            pbar.update(len(batch))

    print(f"✓ {total_inserted:,} sentences inserted into {TABLE}")

# ====================================================
# MAIN
# ====================================================

def main():

    conn = psycopg2.connect(**DB)

    if RESTART:
        reset_table(conn)
        if SENTENCES_CHECKPOINT.exists():
            SENTENCES_CHECKPOINT.unlink()
            print(f"⚠ Deleted checkpoint {SENTENCES_CHECKPOINT}")

    # ── Load models ──────────────────────────────────
    print("Loading NLP + embedding models...")
    nlp = spacy.load("en_core_web_sm")

    sentence_extractor = SentenceExtractor(nlp, batch_size=2000, n_process=1)

    model = SentenceTransformer(MODEL_NAME)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"✓ Embedding dimension: {embed_dim}")

    preprocessor = Preprocessor(
        sentence_extractor=sentence_extractor,
        model=model,
        encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS, 
            "show_progress_bar": False,
            "batch_size": BATCH_SIZE
        }
    )

    setup_table(conn, embed_dim)

    # ── Phase 1: Segmentation ────────────────────────
    sentences_df = run_segmentation(preprocessor)

    # ── Phase 2: Embed + insert ──────────────────────
    run_embedding(conn, sentences_df, preprocessor)

    # ── Build index ──────────────────────────────────
    if BUILD_INDEX:
        build_hnsw_index(conn)

    conn.close()
    print("✅ Pipeline complete")


# ====================================================
if __name__ == "__main__":
    mp.freeze_support()
    main()