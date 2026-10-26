# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# pylint: disable=no-member

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import psycopg 
from pgvector.psycopg import register_vector

from src.config import settings

@contextmanager
def connect() -> Generator[psycopg.Connection, None, None]:
    conn: psycopg.Connection = psycopg.connect(settings.database_url)
    register_vector(conn)
    try: 
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def wait_for_db(attempts: int = 30, delay: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
                return
        except psycopg.OperationalError as exc:
            last_error = exc
            time.sleep(delay)
    raise RuntimeError(
        f"database not reachable after {attempts} attempts: {settings.database_url}"
    ) from last_error