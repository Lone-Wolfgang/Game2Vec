import pandas as pd
import psycopg2
import psycopg2.extras

# ====================================================
# Database connection
# ====================================================

DB = dict(
    dbname="steam_rec",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433,
)

conn = psycopg2.connect(**DB)

# ====================================================
# Load usertags
# ====================================================

with conn.cursor() as cur:
    cur.execute("SELECT usertag_id, usertag FROM usertags;")
    usertags = pd.DataFrame(cur.fetchall(), columns=["usertag_id", "usertag"])

# ====================================================
# Load and process application-tag associations
# ====================================================

usertags_applications = pd.read_parquet(
    '/Users/jwkle/Documents/Steam/project/data/scraped/assembled/tags.parquet'
)

# Explode lists into individual rows
usertags_applications = usertags_applications.explode('user_tags')
usertags_applications.columns = ['app_id', 'usertag']

# Compute ranks and votes
usertags_applications['tagrank'] = usertags_applications.groupby('app_id').cumcount() + 1
usertags_applications['votes'] = usertags_applications['tagrank'].apply(lambda x: int(1.1 ** (100 - x)))

# Merge to get usertag_id
usertags_applications = usertags_applications.merge(usertags, on='usertag')
usertags_applications = usertags_applications[['app_id', 'usertag_id', 'votes']]

# ====================================================
# Create applications_usertags table
# ====================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.applications_usertags (
    app_id INTEGER NOT NULL,
    usertag_id INTEGER NOT NULL,
    votes INTEGER,
    PRIMARY KEY (app_id, usertag_id),
    FOREIGN KEY (app_id) REFERENCES public.applications(app_id) ON DELETE CASCADE,
    FOREIGN KEY (usertag_id) REFERENCES public.usertags(usertag_id) ON DELETE CASCADE
);
"""

with conn.cursor() as cur:
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

print("✓ applications_usertags table created")

# ====================================================
# Insert data into applications_usertags
# ====================================================

def insert_applications_usertags(df):
    values = [
        (int(row[0]), int(row[1]), int(row[2])) for row in df.to_numpy()
    ]
    BATCH_SIZE = 1000

    with conn.cursor() as cur:
        for i in range(0, len(values), BATCH_SIZE):
            batch = values[i:i + BATCH_SIZE]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO public.applications_usertags (app_id, usertag_id, votes)
                VALUES %s
                ON CONFLICT (app_id, usertag_id) DO UPDATE
                SET votes = EXCLUDED.votes
                """,
                batch,
                template="(%s,%s,%s)"
            )
        conn.commit()

insert_applications_usertags(usertags_applications)
print("✓ Data inserted into applications_usertags")

conn.close()