"""
Numpy-backed vector store for match documents.

Replaces ChromaDB (incompatible with Python 3.14) with a simple
persistent store: OpenAI embeddings stored as .npy + JSON sidecar.

Handles:
  - One-time ingestion of MatchDocument objects (upsert-safe)
  - Semantic + metadata-filtered retrieval for RAG queries
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from backend.rag.document_builder import MatchDocument

_DEFAULT_PERSIST_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "vector_store"
)
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBED_BATCH_SIZE = 512  # OpenAI allows up to 2048; keep conservative


class MatchVectorStore:
    """
    Persistent vector store backed by numpy + JSON.

    Embeddings are produced by OpenAI's text-embedding-3-small model and
    saved to disk so ingestion only needs to happen once (or on data updates).
    Retrieval uses cosine similarity computed in-memory with numpy.
    """

    def __init__(self, persist_directory: str | Path | None = None) -> None:
        self._path = Path(persist_directory) if persist_directory else _DEFAULT_PERSIST_DIR
        self._path.mkdir(parents=True, exist_ok=True)

        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        # In-memory state
        self._embeddings: np.ndarray | None = None  # shape (N, D)
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _emb_path(self) -> Path:
        return self._path / "embeddings.npy"

    def _meta_path(self) -> Path:
        return self._path / "meta.json"

    def _save(self) -> None:
        if self._embeddings is not None:
            np.save(str(self._emb_path()), self._embeddings)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(
                {"ids": self._ids, "documents": self._documents, "metadatas": self._metadatas},
                f,
            )

    def _load(self) -> None:
        if self._emb_path().exists() and self._meta_path().exists():
            self._embeddings = np.load(str(self._emb_path()))
            with open(self._meta_path(), encoding="utf-8") as f:
                data = json.load(f)
            self._ids = data["ids"]
            self._documents = data["documents"]
            self._metadatas = data["metadatas"]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI embeddings API and return (N, D) float32 array."""
        response = self._client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=texts,
        )
        return np.array([e.embedding for e in response.data], dtype=np.float32)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[MatchDocument], batch_size: int = _EMBED_BATCH_SIZE) -> int:
        """
        Upsert documents into the store.
        Documents with existing IDs are updated in-place; new ones are appended.
        Returns the number of documents processed.
        """
        if not documents:
            return 0

        existing_id_index: dict[str, int] = {doc_id: i for i, doc_id in enumerate(self._ids)}

        to_update: list[tuple[int, MatchDocument]] = []
        to_add: list[MatchDocument] = []

        for doc in documents:
            if doc.doc_id in existing_id_index:
                to_update.append((existing_id_index[doc.doc_id], doc))
            else:
                to_add.append(doc)

        # Update existing entries
        if to_update:
            texts = [doc.text for _, doc in to_update]
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_embs = self._embed(batch_texts)
                for j, (idx, doc) in enumerate(to_update[start : start + batch_size]):
                    self._embeddings[idx] = batch_embs[j]
                    self._documents[idx] = doc.text
                    self._metadatas[idx] = doc.metadata

        # Add new entries
        if to_add:
            new_embs_list: list[np.ndarray] = []
            for start in range(0, len(to_add), batch_size):
                batch = to_add[start : start + batch_size]
                new_embs_list.append(self._embed([d.text for d in batch]))

            new_embs = np.vstack(new_embs_list)
            if self._embeddings is None:
                self._embeddings = new_embs
            else:
                self._embeddings = np.vstack([self._embeddings, new_embs])

            self._ids.extend(d.doc_id for d in to_add)
            self._documents.extend(d.text for d in to_add)
            self._metadatas.extend(d.metadata for d in to_add)

        self._save()
        return len(documents)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _cosine_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """Return cosine similarity of query_vec against all stored embeddings."""
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        return (self._embeddings / norms) @ q

    def _apply_filter(self, where: dict, metadata: dict) -> bool:
        """Evaluate a Chroma-style metadata filter against a single document's metadata."""
        for key, condition in where.items():
            if key == "$or":
                if not any(self._apply_filter(clause, metadata) for clause in condition):
                    return False
            elif key == "$and":
                if not all(self._apply_filter(clause, metadata) for clause in condition):
                    return False
            elif isinstance(condition, dict):
                field_val = metadata.get(key)
                for op, operand in condition.items():
                    if op == "$eq" and field_val != operand:
                        return False
                    elif op == "$ne" and field_val == operand:
                        return False
                    elif op == "$in" and field_val not in operand:
                        return False
                    elif op == "$nin" and field_val in operand:
                        return False
            else:
                # Equality shorthand: {"field": value}
                if metadata.get(key) != condition:
                    return False
        return True

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically similar match documents.

        Returns a list of dicts with keys: id, text, metadata, distance.
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        query_vec = self._embed([query_text])[0]
        sims = self._cosine_similarities(query_vec)

        # Build mask from metadata filter
        if where:
            mask = np.array(
                [self._apply_filter(where, meta) for meta in self._metadatas],
                dtype=bool,
            )
        else:
            mask = np.ones(len(self._ids), dtype=bool)

        masked_sims = np.where(mask, sims, -np.inf)
        top_n = min(n_results, int(mask.sum()))
        top_indices = np.argpartition(masked_sims, -top_n)[-top_n:] if top_n > 0 else []
        top_indices = sorted(top_indices, key=lambda i: masked_sims[i], reverse=True)

        return [
            {
                "id": self._ids[i],
                "text": self._documents[i],
                "metadata": self._metadatas[i],
                "distance": float(1.0 - sims[i]),
            }
            for i in top_indices
            if masked_sims[i] > -np.inf
        ]

    def query_for_teams(
        self,
        query_text: str,
        team_ids: list[str],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve matches involving any of the given team IDs,
        ranked by semantic similarity to query_text.
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
