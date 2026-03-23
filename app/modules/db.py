"""
modules/db.py
=============
Thin wrapper around a psycopg2 PostgreSQL connection.

Exposes a single method used by the rest of the codebase:
    execute_query(sql, params) → pd.DataFrame
"""

import pandas as pd
import psycopg2
from typing import Dict, Optional


class Database:
    """Manages a single persistent psycopg2 connection."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        connect_on_init: bool = True,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.schema: Dict[str, pd.DataFrame] | None = None
        if connect_on_init:
            self.connect()

    def connect(self) -> None:
        """Open the connection and cache the database schema."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            self.schema = self._fetch_schemas()
            print(f"Connected to {self.database} at {self.host}:{self.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connection = None
            self.schema = None

    def close(self) -> None:
        """Close the connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Run a SELECT query and return results as a DataFrame."""
        if not self.connection:
            raise ValueError("No active database connection.")
        return pd.read_sql_query(query, self.connection, params=params)

    def _fetch_schemas(self) -> Dict[str, pd.DataFrame]:
        """Return column metadata for every table in the public schema."""
        sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """
        cursor = self.connection.cursor()
        cursor.execute(sql)
        tables = [row[0] for row in cursor.fetchall()]

        schemas = {}
        for table in tables:
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default,
                       character_maximum_length
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
                """,
                (table,),
            )
            schemas[table] = pd.DataFrame(
                cursor.fetchall(),
                columns=["column_name", "data_type", "is_nullable",
                         "column_default", "character_maximum_length"],
            )
        cursor.close()
        return schemas