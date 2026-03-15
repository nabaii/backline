"""MongoDB connection management for Backline.

Provides a lazily-initialised MongoClient and convenience accessors for the
default database and its collections.  The connection string is read from
the ``MONGODB_URI`` environment variable.

When ``MONGODB_URI`` is not set the module exposes ``None`` values so that
callers can fall back to local CSV loading without crashing on import.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

_client = None
_db = None


def _get_client():
    global _client
    if _client is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            return None
        from pymongo import MongoClient
        _client = MongoClient(uri)
    return _client


def get_db() -> "Database | None":
    """Return the default Backline database, or ``None`` when unconfigured."""
    global _db
    if _db is None:
        client = _get_client()
        if client is None:
            return None
        _db = client.get_default_database("backline")
    return _db


def get_collection(name: str) -> "Collection | None":
    """Return a collection handle, or ``None`` when MongoDB is unavailable."""
    db = get_db()
    if db is None:
        return None
    return db[name]
