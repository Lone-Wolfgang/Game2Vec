import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from psycopg2 import sql
from tqdm import tqdm

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
TABLE = "applications"

ID_COL = "app_id"
TEXT_COL = "short_description"
EMBED_COL = "description_embedding"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
NORMALIZE_EMBEDDINGS = True

BATCH_SIZE = 128
OVERWRITE_EMBEDDINGS = True

# ============================================================
# MODEL
# ============================================================

print("✓ Loading model...")
model = SentenceTransformer(MODEL_NAME)
EMBED_DIM = model.get_sentence_embedding_dimension()
print(f"✓ Embedding dimension: {EMBED_DIM}")

# ============================================================
# DATABASE
# ============================================================

def connect():
    return psycopg2.connect(**DB)


def ensure_vector_column(conn):
    """
    Recreate vector column if dimension changed.
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
            existing_dim = result[0] - 4  # pgvector stores dim + 4

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


def ensure_hnsw_index(conn):
    """
    Create HNSW index if missing.
    """
    index_name = f"{TABLE}_{EMBED_COL}_hnsw"

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname=%s
            AND tablename=%s
            AND indexname=%s
            """,
            (SCHEMA, TABLE, index_name),
        )

        if cur.fetchone():
            print("✓ HNSW index already exists")
            return

        print("✓ Creating HNSW index")

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX {}
                ON {}.{}
                USING hnsw ({} vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
                """
            ).format(
                sql.Identifier(index_name),
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE),
                sql.Identifier(EMBED_COL),
            )
        )

    conn.commit()


# ============================================================
# DATA
# ============================================================

def fetch_rows(conn):
    cur = conn.cursor(name="embed_cursor")
    cur.execute(
        sql.SQL(
            """
            SELECT {}, {}
            FROM {}.{}
            WHERE {} IS NOT NULL
            """
        ).format(
            sql.Identifier(ID_COL),
            sql.Identifier(TEXT_COL),
            sql.Identifier(SCHEMA),
            sql.Identifier(TABLE),
            sql.Identifier(TEXT_COL),
        )
    )
    return cur


def count_rows(conn):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM {}.{}
                WHERE {} IS NOT NULL
                """
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE),
                sql.Identifier(TEXT_COL),
            )
        )
        return cur.fetchone()[0]


def update_batch(conn, rows):
    ids, texts = zip(*rows)

    embeddings = model.encode(
        list(texts),
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=False,
    )

    values = [(int(i), e.tolist()) for i, e in zip(ids, embeddings)]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            sql.SQL(
                """
                UPDATE {}.{} AS t
                SET {} = v.embedding::vector
                FROM (VALUES %s) AS v(id, embedding)
                WHERE t.{} = v.id
                """
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE),
                sql.Identifier(EMBED_COL),
                sql.Identifier(ID_COL),
            ),
            values,
            template="(%s, %s)",
        )

    conn.commit()


# ============================================================
# MAIN
# ============================================================

def main():
    read_conn = connect()
    write_conn = connect()

    ensure_vector_column(write_conn)

    if OVERWRITE_EMBEDDINGS:

        total = count_rows(read_conn)

        if total == 0:
            print("✓ No rows to embed")
            return

        cur = fetch_rows(read_conn)
        batch = []

        with tqdm(total=total, desc="Embedding", unit="rows") as pbar:
            for row in cur:
                batch.append(row)

                if len(batch) >= BATCH_SIZE:
                    update_batch(write_conn, batch)
                    pbar.update(len(batch))
                    batch.clear()

            if batch:
                update_batch(write_conn, batch)
                pbar.update(len(batch))

        cur.close()
        print("✓ Embeddings written")

    else:
        print("✓ Skipping embedding computation")

    ensure_hnsw_index(write_conn)

    read_conn.close()
    write_conn.close()


if __name__ == "__main__":
    main()