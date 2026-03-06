"""
RAG pipeline: query → retrieve → generate.

Flow
----
1. Accept a natural-language user query (and optional team context).
2. Retrieve the most relevant match documents from ChromaDB.
3. Build a prompt with those documents as context.
4. Call OpenAI GPT and stream the response back.

The pipeline is intentionally stateless — call ``run()`` per request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Generator

from openai import OpenAI

from backend.rag.vector_store import MatchVectorStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_N_RESULTS = 12        # matches to pull from Chroma
_MAX_CONTEXT_CHARS = 20_000    # safety cap before sending to the LLM

_SYSTEM_PROMPT = """\
You are a football analytics assistant for the Backline platform.
You have been given a set of real historical match records retrieved from a \
vector database. Each record contains statistics for a specific match.

Your job is to help the user understand tactical trends, hit rates, and \
patterns in the data. Be specific — reference actual match stats from the \
context. Do NOT make predictions or tell the user to bet. Focus on what the \
data shows.

If the context doesn't contain enough information to answer the question, say \
so clearly rather than speculating."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RAGRequest:
    query: str
    home_team_id: str | None = None
    away_team_id: str | None = None
    n_results: int = _DEFAULT_N_RESULTS
    model: str = _DEFAULT_MODEL
    extra_context: str = ""             # e.g. upcoming fixture info


@dataclass
class RAGResponse:
    answer: str
    retrieved_matches: list[dict] = field(default_factory=list)
    match_count: int = 0


class RAGPipeline:
    """
    Orchestrates document retrieval and LLM generation.

    Parameters
    ----------
    vector_store:
        An already-initialised MatchVectorStore.
    api_key:
        OpenAI API key. Falls back to the ``OPENAI_API_KEY``
        environment variable if not supplied.
    """

    def __init__(
        self,
        vector_store: MatchVectorStore,
        api_key: str | None = None,
    ) -> None:
        self._store = vector_store
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieve_and_build_prompt(self, request: RAGRequest) -> tuple[list[dict], str]:
        """Retrieve hits and return (hits, user_message)."""
        team_ids = [t for t in [request.home_team_id, request.away_team_id] if t]
        if team_ids:
            hits = self._store.query_for_teams(
                request.query,
                team_ids=team_ids,
                n_results=request.n_results,
            )
        else:
            hits = self._store.query(request.query, n_results=request.n_results)

        context_parts: list[str] = []
        char_budget = _MAX_CONTEXT_CHARS
        for hit in hits:
            snippet = hit["text"]
            if len(snippet) > char_budget:
                break
            context_parts.append(snippet)
            char_budget -= len(snippet)

        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No matching matches found."

        fixture_line = ""
        if request.home_team_id and request.away_team_id:
            fixture_line = (
                f"Upcoming fixture: {request.home_team_id} (home) "
                f"vs {request.away_team_id} (away).\n\n"
            )

        extra = f"{request.extra_context}\n\n" if request.extra_context else ""
        user_message = (
            f"{fixture_line}{extra}"
            f"Historical match context:\n\n{context_block}\n\n"
            f"User question: {request.query}"
        )

        return hits, user_message

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def run(self, request: RAGRequest) -> RAGResponse:
        """Retrieve relevant matches and ask GPT to analyse them."""
        hits, user_message = self._retrieve_and_build_prompt(request)

        response = self._client.chat.completions.create(
            model=request.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer = response.choices[0].message.content or ""

        return RAGResponse(
            answer=answer,
            retrieved_matches=hits,
            match_count=len(hits),
        )

    def stream(self, request: RAGRequest) -> Generator[str, None, None]:
        """
        Same as ``run()`` but yields text chunks as they arrive from GPT.
        Useful for server-sent events on the Flask side.
        """
        hits, user_message = self._retrieve_and_build_prompt(request)

        response = self._client.chat.completions.create(
            model=request.model,
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
