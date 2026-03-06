"""
ChromaDB-backed vector store for match documents.

Handles:
  - One-time ingestion of MatchDocument objects (upsert-safe)
  - Semantic + metadata-filtered retrieval for RAG queries
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.rag.document_builder import MatchDocument

# Persist the vector DB alongside the data directory
_DEFAULT_PERSIST_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
)
_COLLECTION_NAME = "match_documents"


class MatchVectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Uses ChromaDB's default embedding function
    (sentence-transformers/all-MiniLM-L6-v2) so no external API key
    is needed for embedding.
    """

    def __init__(self, persist_directory: str | Path | None = None) -> None:
        path = Path(persist_directory) if persist_directory else _DEFAULT_PERSIST_DIR
        path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[MatchDocument], batch_size: int = 100) -> int:
        """
        Upsert documents into the collection.

        Returns the number of documents processed.
        """
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            self._collection.upsert(
                ids=[d.doc_id for d in batch],
                documents=[d.text for d in batch],
                metadatas=[d.metadata for d in batch],
            )
            total += len(batch)

        return total

    @property
    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically similar match documents.

        Parameters
        ----------
        query_text:
            Natural-language query from the user.
        n_results:
            Maximum number of matches to return.
        where:
            Optional Chroma metadata filter dict, e.g.
            ``{"home_team": {"$in": ["liverpool", "arsenal"]}}``.

        Returns
        -------
        List of dicts, each with keys: ``id``, ``text``, ``metadata``,
        ``distance``.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "distance": dist,
                }
            )

        return hits

    def query_for_teams(
        self,
        query_text: str,
        team_ids: list[str],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve matches that involve any of the given team IDs,
        ranked by semantic similarity to ``query_text``.

        Uses a Chroma ``$or`` filter so that both home and away
        appearances are captured.
        """
        if not team_ids:
            return self.query(query_text, n_results=n_results)

        where: dict[str, Any] = {
            "$or": [
                {"home_team_id": {"$in": team_ids}},
                {"away_team_id": {"$in": team_ids}},
            ]
        }
        return self.query(query_text, n_results=n_results, where=where)
