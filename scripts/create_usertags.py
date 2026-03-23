import json
import time
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from psycopg2 import sql

# ============================================================
# CONFIG
# ============================================================

DB = dict(
    dbname="steam_rec",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433,
)

SCHEMA = "public"
TABLE = "usertags"

ID_COL = "usertag_id"
TAG_COL = "usertag"
DESC_COL = "usertag_description"

EMBED_COL = "usertag_embedding"
EMBED_SOURCE_COL = TAG_COL

MODEL_NAME = "BAAI/bge-base-en-v1.5"
NORMALIZE_EMBEDDINGS = True

BATCH_SIZE = 256

JSON_PATH = Path(
    "/Users/jwkle/Documents/Steam/project/data/tag_description_expanded.json"
)

# ============================================================
# MODEL
# ============================================================

print("✓ Loading model...")
model = SentenceTransformer(MODEL_NAME)
EMBED_DIM = model.get_sentence_embedding_dimension()
print(f"✓ Embedding dim: {EMBED_DIM}")

# ============================================================
# DATABASE
# ============================================================

def connect():
    return psycopg2.connect(**DB)


# ============================================================
# SCHEMA
# ============================================================

def ensure_table(conn):
    """
    Create usertags table if it doesn't exist, with all required constraints.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    {} SERIAL PRIMARY KEY,
                    {} VARCHAR NOT NULL UNIQUE,
                    {} VARCHAR,
                    {} vector(%s)
                )
                """
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE),
                sql.Identifier(ID_COL),
                sql.Identifier(TAG_COL),
                sql.Identifier(DESC_COL),
                sql.Identifier(EMBED_COL),
            ),
            (EMBED_DIM,),
        )
        # Add unique constraint to existing table if missing
        cur.execute(
            sql.SQL(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = %s::regclass
                        AND contype = 'u'
                        AND conname = %s
                    ) THEN
                        ALTER TABLE {}.{} ADD CONSTRAINT {} UNIQUE ({});
                    END IF;
                END$$;
                """
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE),
                sql.Identifier(f"{TABLE}_{TAG_COL}_unique"),
                sql.Identifier(TAG_COL),
            ),
            (f"{SCHEMA}.{TABLE}", f"{TABLE}_{TAG_COL}_unique"),
        )
    conn.commit()
    print(f"✓ Table {TABLE} ready")


def ensure_embedding_column(conn):
    """
    Create embedding column if missing.
    Recreate if dimension mismatch.
    """
    with conn.cursor() as cur:

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = %s::regclass
            AND attname = %s
            """,
            (f"{SCHEMA}.{TABLE}", EMBED_COL),
        )

        result = cur.fetchone()

        if result is None:

            print("✓ Creating embedding column")

            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN {} vector(%s)"
                ).format(
                    sql.Identifier(SCHEMA),
                    sql.Identifier(TABLE),
                    sql.Identifier(EMBED_COL),
                ),
                (EMBED_DIM,),
            )

        else:

            existing_dim = result[0] - 4

            if existing_dim != EMBED_DIM:

                print(f"⚠ Dimension mismatch ({existing_dim} → {EMBED_DIM})")
                print("⚠ Recreating embedding column")

                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {}.{} DROP COLUMN {}"
                    ).format(
                        sql.Identifier(SCHEMA),
                        sql.Identifier(TABLE),
                        sql.Identifier(EMBED_COL),
                    )
                )

                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {}.{} ADD COLUMN {} vector(%s)"
                    ).format(
                        sql.Identifier(SCHEMA),
                        sql.Identifier(TABLE),
                        sql.Identifier(EMBED_COL),
                    ),
                    (EMBED_DIM,),
                )

    conn.commit()


# ============================================================
# DATA LOADING
# ============================================================

def load_tags() -> pd.DataFrame:

    raw = json.loads(JSON_PATH.read_text())

    merged = {}
    for block in raw:
        merged.update(block)

    df = pd.DataFrame(
        merged.items(),
        columns=[TAG_COL, DESC_COL],
    )

    return df


# ============================================================
# EMBEDDING
# ============================================================

def embed_column(df: pd.DataFrame) -> pd.DataFrame:

    texts = df[EMBED_SOURCE_COL].fillna("").tolist()

    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts)):,}/{len(texts):,}", flush=True)

    df[EMBED_COL] = all_embeddings

    return df


# ============================================================
# DATABASE UPDATE
# ============================================================

def update_embeddings(conn, df: pd.DataFrame, batch_size=500):

    total = len(df)
    start = time.time()

    with conn.cursor() as cur:

        for i in range(0, total, batch_size):

            chunk = df.iloc[i : i + batch_size]

            values = [
                (row[TAG_COL], row[DESC_COL], row[EMBED_COL])
                for _, row in chunk.iterrows()
            ]

            psycopg2.extras.execute_values(
                cur,
                sql.SQL(
                    """
                    INSERT INTO {}.{} ({}, {}, {})
                    VALUES %s
                    ON CONFLICT ({}) DO UPDATE
                        SET {} = EXCLUDED.{},
                            {} = EXCLUDED.{}
                    """
                ).format(
                    sql.Identifier(SCHEMA),
                    sql.Identifier(TABLE),
                    sql.Identifier(TAG_COL),
                    sql.Identifier(DESC_COL),
                    sql.Identifier(EMBED_COL),
                    sql.Identifier(TAG_COL),       # conflict target
                    sql.Identifier(DESC_COL),      # update desc
                    sql.Identifier(DESC_COL),
                    sql.Identifier(EMBED_COL),     # update embedding
                    sql.Identifier(EMBED_COL),
                ),
                values,
                template="(%s, %s, %s)",
            )

            conn.commit()

            print(
                f"Inserted/updated {min(i + batch_size, total):,}/{total:,} "
                f"({time.time() - start:.1f}s)",
                flush=True,
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading tags...")
    df = load_tags()

    print(f"✓ {len(df):,} tags loaded")

    print("Generating embeddings...")
    df = embed_column(df)

    conn = connect()

    ensure_table(conn)
    ensure_embedding_column(conn)

    print("Updating database...")
    update_embeddings(conn, df)

    conn.close()

    print("✓ Embeddings updated successfully")


if __name__ == "__main__":
    main()