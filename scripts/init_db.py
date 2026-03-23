import psycopg2
import psycopg2.extras
from psycopg2 import sql
from pathlib import Path

# =========================
# CONFIG
# =========================

SOURCE_DB = {
    "dbname": "steam_dataset",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5433,
}

ADMIN_DB = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5433,
}

TARGET_DB_NAME = "steam_rec"

SCHEMA = "public"
TABLE_NAME = "applications"
SOURCE_ID_COLUMN = "appid"       # column name in the source DB
TARGET_ID_COLUMN = "app_id"      # column name in the target DB
EXCLUDE_COLUMNS = {"description_embedding"}

APPIDS = set(
    map(
        int,
        Path(
            "/Users/jwkle/Documents/Steam/project/data/scraped/assembled/appids.txt"
        ).read_text().splitlines()
    )
)

# =========================
# HELPERS
# =========================

def connect(cfg):
    return psycopg2.connect(**cfg)

def create_target_db():
    conn = connect(ADMIN_DB)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TARGET_DB_NAME,))
            if cur.fetchone():
                print(f"✓ Database {TARGET_DB_NAME} already exists — skipping creation")
                return
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TARGET_DB_NAME)))
            print(f"✓ Created database {TARGET_DB_NAME}")
    finally:
        conn.close()

def clone_user_defined_types():
    with connect(SOURCE_DB) as src_conn, src_conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.nspname, t.typname, e.enumlabel
            FROM pg_type t
            JOIN pg_namespace n ON n.oid = t.typnamespace
            JOIN pg_enum e ON e.enumtypid = t.oid
            WHERE t.typtype='e' AND n.nspname=%s
            ORDER BY t.typname, e.enumsortorder
            """,
            (SCHEMA,),
        )
        rows = cur.fetchall()

    if not rows:
        print("✓ No enum types to clone")
        return

    enums = {}
    for schema, name, label in rows:
        enums.setdefault((schema, name), []).append(label)

    with connect({**SOURCE_DB, "dbname": TARGET_DB_NAME}) as tgt_conn, tgt_conn.cursor() as cur:
        for (schema, name), labels in enums.items():
            labels_sql = ", ".join(f"'{l}'" for l in labels)
            cur.execute(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_type t
                        JOIN pg_namespace n ON n.oid = t.typnamespace
                        WHERE t.typname = %s AND n.nspname = %s
                    ) THEN
                        CREATE TYPE "{schema}"."{name}" AS ENUM ({labels_sql});
                    END IF;
                END
                $$;
                """,
                (name, schema),
            )
        tgt_conn.commit()
    print("✓ Cloned enum types")

def clone_table_schema():
    with connect(SOURCE_DB) as src_conn, src_conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, udt_schema, udt_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
            """,
            (SCHEMA, TABLE_NAME),
        )
        columns = cur.fetchall()

    column_defs = []
    for name, udt_schema, udt_name, nullable, default in columns:
        if name in EXCLUDE_COLUMNS:
            continue
        # Rename appid -> app_id in the target schema
        target_name = TARGET_ID_COLUMN if name == SOURCE_ID_COLUMN else name
        col = f'"{target_name}" "{udt_schema}"."{udt_name}"'
        if default:
            col += f" DEFAULT {default}"
        if nullable == "NO":
            col += " NOT NULL"
        column_defs.append(col)

    create_sql = f'''
        CREATE TABLE IF NOT EXISTS "{SCHEMA}"."{TABLE_NAME}" (
            {", ".join(column_defs)}
        )
    '''

    with connect({**SOURCE_DB, "dbname": TARGET_DB_NAME}) as tgt_conn, tgt_conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
        cur.execute(create_sql)
        tgt_conn.commit()
    print(f"✓ Cloned schema for table {TABLE_NAME} (appid → app_id)")

def ensure_primary_key():
    """Ensure app_id is PRIMARY KEY in applications table"""
    with connect({**SOURCE_DB, "dbname": TARGET_DB_NAME}) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_schema=%s AND table_name=%s AND constraint_type='PRIMARY KEY'
            """,
            (SCHEMA, TABLE_NAME),
        )
        if cur.fetchone():
            print(f"✓ Primary key already exists on {TABLE_NAME}.{TARGET_ID_COLUMN}")
            return

        cur.execute(
            sql.SQL(
                "ALTER TABLE {}.{} ADD CONSTRAINT {} PRIMARY KEY ({})"
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE_NAME),
                sql.Identifier(f"{TABLE_NAME}_pk"),
                sql.Identifier(TARGET_ID_COLUMN),
            )
        )
        conn.commit()
        print(f"✓ Added primary key to {TABLE_NAME}.{TARGET_ID_COLUMN}")

def copy_subset():
    with connect(SOURCE_DB) as src_conn, connect({**SOURCE_DB, "dbname": TARGET_DB_NAME}) as tgt_conn:
        src_cur = src_conn.cursor(name="stream_cursor")
        tgt_cur = tgt_conn.cursor()

        # Query source using the source column name
        src_cur.execute(
            sql.SQL("SELECT * FROM {}.{} WHERE {} = ANY(%s)").format(
                sql.Identifier(SCHEMA),
                sql.Identifier(TABLE_NAME),
                sql.Identifier(SOURCE_ID_COLUMN),
            ),
            (list(APPIDS),),
        )

        first_row = src_cur.fetchone()
        if first_row is None:
            print("✓ No rows to copy")
            return

        src_colnames = [d.name for d in src_cur.description]

        # detect json/jsonb columns
        with src_conn.cursor() as meta_cur:
            meta_cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=%s AND table_name=%s AND data_type IN ('json','jsonb')
                """,
                (SCHEMA, TABLE_NAME),
            )
            json_cols = {r[0] for r in meta_cur.fetchall()}

        keep_idx = [i for i, c in enumerate(src_colnames) if c not in EXCLUDE_COLUMNS]

        # Remap source column names to target column names
        keep_cols = [
            TARGET_ID_COLUMN if src_colnames[i] == SOURCE_ID_COLUMN else src_colnames[i]
            for i in keep_idx
        ]

        insert_sql = sql.SQL(
            "INSERT INTO {}.{} ({}) VALUES %s ON CONFLICT DO NOTHING"
        ).format(
            sql.Identifier(SCHEMA),
            sql.Identifier(TABLE_NAME),
            sql.SQL(", ").join(map(sql.Identifier, keep_cols)),
        )

        def project(row):
            out = []
            for i in keep_idx:
                val = row[i]
                if src_colnames[i] in json_cols and val is not None:
                    val = psycopg2.extras.Json(val)
                out.append(val)
            return tuple(out)

        BATCH_SIZE = 1000
        batch = [project(first_row)]
        total = 0

        for row in src_cur:
            batch.append(project(row))
            if len(batch) >= BATCH_SIZE:
                psycopg2.extras.execute_values(tgt_cur, insert_sql, batch)
                total += len(batch)
                batch.clear()

        if batch:
            psycopg2.extras.execute_values(tgt_cur, insert_sql, batch)
            total += len(batch)

        tgt_conn.commit()
        src_cur.close()

    print(f"✓ Copied {total} rows into {TABLE_NAME}")

# =========================
# MAIN
# =========================

def main():
    create_target_db()
    clone_user_defined_types()
    clone_table_schema()
    ensure_primary_key()
    copy_subset()
    print("✓ Initialization complete")

if __name__ == "__main__":
    main()