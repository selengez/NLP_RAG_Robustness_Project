# Dense retriever using FAISS. Supports source filtering and optional reranking.

from dataclasses import dataclass

import numpy as np

from src.indexing.indexer import ChunkMetadata, DocumentIndexer


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its score and source metadata."""
    rank:        int
    text:        str
    score:       float
    doc_id:      str
    chunk_index: int
    source_type: str
    poison_type: str


class Retriever:
    """FAISS-backed dense retriever with optional source filtering and reranking."""

    def __init__(self, indexer, top_k=5, rerank=False, rerank_factor=3, diverse=True):
        self._indexer      = indexer
        self.top_k         = top_k
        self.rerank        = rerank
        self.rerank_factor = rerank_factor
        self.diverse       = diverse
        # Reuse the model already loaded by the indexer
        self._model        = indexer._model

    def retrieve(self, query, source_filter=None):
        """Return top-k chunks most relevant to the query."""
        query_vec = self._encode(query)

        # Decide how many candidates to fetch from FAISS
        if source_filter is not None:
            fetch_k = self.top_k * 10
        elif self.rerank:
            fetch_k = self.top_k * self.rerank_factor
        else:
            fetch_k = self.top_k

        fetch_k = min(fetch_k, self._indexer.index.ntotal)
        distances, indices = self._indexer.index.search(query_vec, fetch_k)
        distances = distances[0]
        indices   = indices[0]

        # Convert FAISS results to (score, metadata) pairs
        candidates = []
        for dist, idx in zip(distances, indices):
            if idx < 0:
                continue
            meta  = self._indexer.metadata[idx]
            score = self._l2_to_cosine(float(dist))
            candidates.append((score, meta))

        # Keep only chunks matching the requested source type
        if source_filter is not None:
            filtered = []
            for s, m in candidates:
                if m.source_type == source_filter:
                    filtered.append((s, m))
            candidates = filtered

        if self.rerank and candidates:
            candidates = self._rerank(query_vec, candidates)

        if self.diverse and candidates:
            candidates = self._diversify(candidates)

        # Build final RetrievalResult list from top-k candidates
        results = []
        for i, (score, meta) in enumerate(candidates[:self.top_k]):
            results.append(RetrievalResult(
                rank        = i + 1,
                text        = meta.text,
                score       = round(score, 4),
                doc_id      = meta.doc_id,
                chunk_index = meta.chunk_index,
                source_type = meta.source_type,
                poison_type = meta.poison_type,
            ))
        return results

    def _rerank(self, query_vec, candidates):
        """Re-score candidates by exact cosine similarity and re-sort."""
        # Collect texts for re-encoding
        texts = []
        for _, meta in candidates:
            texts.append(meta.text)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy  = True,
            show_progress_bar = False,
        ).astype(np.float32)

        # Compute cosine similarity between query and each candidate
        q_norm = query_vec[0] / (np.linalg.norm(query_vec[0]) + 1e-9)
        scores = (embeddings @ q_norm).tolist()

        # Pair scores with metadata and sort by descending score
        metas = []
        for _, meta in candidates:
            metas.append(meta)

        reranked = sorted(
            zip(scores, metas),
            key     = lambda x: x[0],
            reverse = True,
        )
        return reranked

    def _diversify(self, candidates):
        """Prefer chunks from different source documents."""
        seen_docs = set()
        primary   = []
        overflow  = []

        for score, meta in candidates:
            if meta.doc_id not in seen_docs:
                seen_docs.add(meta.doc_id)
                primary.append((score, meta))
            else:
                overflow.append((score, meta))

        return primary + overflow

    def _encode(self, text):
        """Encode a query string to a float32 embedding."""
        return self._model.encode(
            [text], convert_to_numpy=True
        ).astype(np.float32)

    @staticmethod
    def _l2_to_cosine(l2_squared):
        """Convert FAISS squared-L2 distance to cosine similarity."""
        return float(np.clip(1.0 - (l2_squared / 2.0), 0.0, 1.0))
