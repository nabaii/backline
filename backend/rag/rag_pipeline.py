"""
RAG pipeline: query → retrieve → generate.

Flow
----
1. Accept a natural-language user query (and optional team context).
2. Retrieve the most relevant match documents from ChromaDB.
3. Build a prompt with those documents as context.
4. Call Claude via the Anthropic SDK and stream the response back.

The pipeline is intentionally stateless — call ``run()`` per request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Generator

import anthropic

from backend.rag.vector_store import MatchVectorStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_N_RESULTS = 12        # matches to pull from Chroma
_MAX_CONTEXT_CHARS = 20_000    # safety cap before sending to Claude

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
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if not supplied.
    """

    def __init__(
        self,
        vector_store: MatchVectorStore,
        api_key: str | None = None,
    ) -> None:
        self._store = vector_store
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def run(self, request: RAGRequest) -> RAGResponse:
        """Retrieve relevant matches and ask Claude to analyse them."""

        # 1. Retrieve
        team_ids = [
            t for t in [request.home_team_id, request.away_team_id] if t
        ]
        if team_ids:
            hits = self._store.query_for_teams(
                request.query,
                team_ids=team_ids,
                n_results=request.n_results,
            )
        else:
            hits = self._store.query(request.query, n_results=request.n_results)

        # 2. Build context string
        context_parts: list[str] = []
        char_budget = _MAX_CONTEXT_CHARS
        for hit in hits:
            snippet = hit["text"]
            if len(snippet) > char_budget:
                break
            context_parts.append(snippet)
            char_budget -= len(snippet)

        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No matching matches found."

        # 3. Build user message
        fixture_line = ""
        if request.home_team_id and request.away_team_id:
            fixture_line = (
                f"Upcoming fixture: {request.home_team_id} (home) "
                f"vs {request.away_team_id} (away).\n\n"
            )

        user_message = (
            f"{fixture_line}"
            f"{request.extra_context}\n\n" if request.extra_context else f"{fixture_line}"
        ) + (
            f"Historical match context:\n\n{context_block}\n\n"
            f"User question: {request.query}"
        )

        # 4. Call Claude
        message = self._client.messages.create(
            model=request.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = message.content[0].text if message.content else ""

        return RAGResponse(
            answer=answer,
            retrieved_matches=hits,
            match_count=len(hits),
        )

    def stream(self, request: RAGRequest) -> Generator[str, None, None]:
        """
        Same as ``run()`` but yields text chunks as they arrive from Claude.
        Useful for server-sent events on the Flask side.
        """
        team_ids = [
            t for t in [request.home_team_id, request.away_team_id] if t
        ]
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

        with self._client.messages.stream(
            model=request.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
