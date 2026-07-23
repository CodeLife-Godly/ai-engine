"""
PostgreSQL connection management.

Provides a single function to create a database connection.
"""

from psycopg import Connection, connect
from psycopg.rows import dict_row

from database.config import config


def get_connection() -> Connection:
    """
    Create and return a PostgreSQL connection.

    Returns:
        psycopg.Connection: Active PostgreSQL connection.
    """

    return connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        row_factory=dict_row,
        autocommit=False,
    )