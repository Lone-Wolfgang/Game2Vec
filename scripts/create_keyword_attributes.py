"""
Migration: add full-text search columns to applications, reviews, and sentences.

For each relation we add:
  - A GENERATED ALWAYS AS ... STORED tsvector column (auto-maintained by Postgres)
  - A GIN index on that column

applications.name also gets a pg_trgm GIN index for fuzzy name matching.

Safe to re-run: all DDL is guarded with IF NOT EXISTS / existence checks.
"""

import psycopg2
from psycopg2 import sql
import threading
import time

# ============================================================
# CONFIG  — match your existing setup
# ============================================================

DB = dict(
    dbname="steam_rec",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433,
)

SCHEMA = "public"

# ============================================================
# HELPERS
# ============================================================

def connect():
    return psycopg2.connect(**DB)


def column_exists(cur, schema: str, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1 FROM pg_attribute
        WHERE attrelid = %s::regclass
          AND attname   = %s
          AND attisdropped = false
        """,
        (f"{schema}.{table}", column),
    )
    return cur.fetchone() is not None


def index_exists(cur, index_name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE indexname = %s",
        (index_name,),
    )
    return cur.fetchone() is not None


# ============================================================
# MIGRATIONS
# ============================================================

MIGRATIONS = [
    # ----------------------------------------------------------
    # applications
    #   tsvector over name + short_description
    #   trigram index on name for fuzzy matching
    # ----------------------------------------------------------
    {
        "description": "applications — tsvector column",
        "table": "applications",
        "column": "search_fts",
        "add_column": """
            ALTER TABLE {schema}.{table}
            ADD COLUMN {column} tsvector
            GENERATED ALWAYS AS (
                to_tsvector(
                    'english',
                    coalesce(name, '') || ' ' || coalesce(short_description, '')
                )
            ) STORED
        """,
        "index_name": "idx_applications_fts",
        "create_index": """
            CREATE INDEX {index_name} ON {schema}.{table}
            USING GIN ({column})
        """,
    },
    {
        "description": "applications — trigram index on name",
        "table": "applications",
        "column": None,
        "index_name": "idx_applications_name_trgm",
        "create_index": """
            CREATE INDEX {index_name} ON {schema}.{table}
            USING GIN (name gin_trgm_ops)
        """,
    },
    # ----------------------------------------------------------
    # reviews
    # ----------------------------------------------------------
    {
        "description": "reviews — tsvector column",
        "table": "reviews",
        "column": "search_fts",
        "add_column": """
            ALTER TABLE {schema}.{table}
            ADD COLUMN {column} tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(review, ''))
            ) STORED
        """,
        "index_name": "idx_reviews_fts",
        "create_index": """
            CREATE INDEX {index_name} ON {schema}.{table}
            USING GIN ({column})
        """,
    },
    # ----------------------------------------------------------
    # sentences
    # ----------------------------------------------------------
    {
        "description": "sentences — tsvector column",
        "table": "sentences",
        "column": "search_fts",
        "add_column": """
            ALTER TABLE {schema}.{table}
            ADD COLUMN {column} tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(sentence, ''))
            ) STORED
        """,
        "index_name": "idx_sentences_fts",
        "create_index": """
            CREATE INDEX {index_name} ON {schema}.{table}
            USING GIN ({column})
        """,
    },
]


# ============================================================
# PROGRESS POLLER
# ============================================================

class ProgressPoller:
    """
    Polls pg_stat_progress_cluster (ALTER TABLE rewrites) and
    pg_stat_progress_create_index (CREATE INDEX) in a background thread,
    printing a progress line every `interval` seconds.
    """

    REWRITE_SQL = """
        SELECT
            phase,
            heap_blks_total,
            heap_blks_scanned
        FROM pg_stat_progress_cluster
        WHERE relid = %s::regclass
    """

    INDEX_SQL = """
        SELECT
            phase,
            blocks_total,
            blocks_done,
            tuples_total,
            tuples_done
        FROM pg_stat_progress_create_index
        WHERE relid = %s::regclass
    """

    def __init__(self, db_config: dict, table: str, interval: float = 5.0):
        self.db_config = db_config
        self.table = table
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self):
        self._stop.clear()
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        print()

    def _poll(self):
        conn = psycopg2.connect(**self.db_config)
        conn.autocommit = True
        start = time.monotonic()
        try:
            while not self._stop.wait(self.interval):
                elapsed = time.monotonic() - start
                with conn.cursor() as cur:
                    # Try index build first (more granular)
                    cur.execute(self.INDEX_SQL, (self.table,))
                    row = cur.fetchone()
                    if row:
                        phase, blk_total, blk_done, tup_total, tup_done = row
                        if tup_total:
                            pct = 100.0 * tup_done / tup_total
                            print(
                                f"\r  [{elapsed:6.0f}s] index build — {phase}"
                                f"  {tup_done:,}/{tup_total:,} tuples ({pct:.1f}%)",
                                end="", flush=True,
                            )
                        elif blk_total:
                            pct = 100.0 * blk_done / blk_total
                            print(
                                f"\r  [{elapsed:6.0f}s] index build — {phase}"
                                f"  {blk_done:,}/{blk_total:,} blocks ({pct:.1f}%)",
                                end="", flush=True,
                            )
                        else:
                            print(
                                f"\r  [{elapsed:6.0f}s] index build — {phase}",
                                end="", flush=True,
                            )
                        continue

                    # Fall back to table rewrite progress
                    cur.execute(self.REWRITE_SQL, (self.table,))
                    row = cur.fetchone()
                    if row:
                        phase, blk_total, blk_scanned = row
                        if blk_total:
                            pct = 100.0 * blk_scanned / blk_total
                            print(
                                f"\r  [{elapsed:6.0f}s] table rewrite — {phase}"
                                f"  {blk_scanned:,}/{blk_total:,} blocks ({pct:.1f}%)",
                                end="", flush=True,
                            )
                        else:
                            print(
                                f"\r  [{elapsed:6.0f}s] table rewrite — {phase}",
                                end="", flush=True,
                            )
                    else:
                        print(
                            f"\r  [{elapsed:6.0f}s] waiting for progress info...",
                            end="", flush=True,
                        )
        finally:
            conn.close()


# ============================================================
# HELPERS
# ============================================================

def fmt(template: str, schema: str, table: str, column: str = None, index_name: str = None) -> str:
    return template.format(
        schema=schema,
        table=table,
        column=column or "",
        index_name=index_name or "",
    )


def apply_performance_settings(cur, conn):
    """
    Session-scoped performance tuning for a dedicated 40GB instance.

    maintenance_work_mem = 8GB
        GIN index builds use this for sorting/accumulation. More memory =
        fewer merge passes = significantly faster index build. 8GB is
        safe when only one migration is running.

    max_parallel_maintenance_workers = 4
        Allows CREATE INDEX to use 4 cores for the sort phase. Re-enabled
        now that we're running intentionally with no competing sessions.
        (Previously disabled to avoid AdminShutdown from killed workers.)

    max_parallel_workers_per_gather = 4
        Parallel workers for the sequential scan during ALTER TABLE rewrite.

    synchronous_commit = off
        Don't wait for WAL flush on every commit. Safe here — worst case
        on crash is we re-run the migration, which is idempotent.

    wal_compression = on
        Compresses WAL records during the table rewrite, reducing write
        I/O volume on the WAL path.
    """
    settings = {
        "maintenance_work_mem":             "8GB",
        "max_parallel_maintenance_workers": "4",
        "max_parallel_workers_per_gather":  "4",
        "synchronous_commit":               "off",
        "wal_compression":                  "on",
    }
    for key, val in settings.items():
        cur.execute(f"SET {key} = %s", (val,))
    conn.commit()

    print("  Performance settings applied:")
    for key, val in settings.items():
        print(f"    {key} = {val}")


# ============================================================
# MIGRATIONS
# ============================================================

def run_migrations(conn):
    with conn.cursor() as cur:

        print("Ensuring extensions...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("  ✓ pg_trgm, vector")

        print("\nApplying performance settings...")
        apply_performance_settings(cur, conn)

        for m in MIGRATIONS:
            table      = m["table"]
            column     = m.get("column")
            index_name = m["index_name"]
            desc       = m["description"]

            print(f"\n[{desc}]")

            # -- Add column if needed ----------------------------------
            if column and "add_column" in m:
                if column_exists(cur, SCHEMA, table, column):
                    print(f"  ✓ Column '{column}' already exists — skipping")
                else:
                    try:
                        ddl = fmt(m["add_column"], SCHEMA, table, column, index_name)
                        with ProgressPoller(DB, table):
                            cur.execute(ddl)
                            conn.commit()
                        print(f"  ✓ Column '{column}' added")
                    except Exception as e:
                        conn.rollback()
                        raise RuntimeError(
                            f"Failed to add column '{column}' on '{table}': {e}"
                        ) from e

            # -- Create index if needed --------------------------------
            if index_exists(cur, index_name):
                print(f"  ✓ Index '{index_name}' already exists — skipping")
            else:
                try:
                    ddl = fmt(m["create_index"], SCHEMA, table, column, index_name)
                    with ProgressPoller(DB, table):
                        cur.execute(ddl)
                        conn.commit()
                    print(f"  ✓ Index '{index_name}' created")
                except Exception as e:
                    conn.rollback()
                    raise RuntimeError(
                        f"Failed to create index '{index_name}' on '{table}': {e}"
                    ) from e


# ============================================================
# MAIN
# ============================================================

def main():
    print("Connecting...")
    conn = connect()
    print("✓ Connected\n")

    try:
        run_migrations(conn)
    finally:
        conn.close()

    print("\n✓ Migration complete")


if __name__ == "__main__":
    main()