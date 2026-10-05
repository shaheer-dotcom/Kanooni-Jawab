"""BAAI/bge-m3 dense + learned sparse embeddings for Qdrant hybrid search."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models as qmodels

from config import Config

log = logging.getLogger(__name__)

_bge_m3: Any | None = None
_dense_dim: int | None = None


def _get_bge_m3():
    global _bge_m3
    if _bge_m3 is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for BAAI/bge-m3. "
                "Install with: pip install FlagEmbedding"
            ) from exc

        log.info("Loading BGE-M3 model: %s", Config.EMBEDDING_MODEL)
        _bge_m3 = BGEM3FlagModel(
            Config.EMBEDDING_MODEL,
            use_fp16=Config.BGE_M3_USE_FP16,
        )
    return _bge_m3


def _encode_bge_m3(
    texts: list[str],
    *,
    return_dense: bool,
    return_sparse: bool,
) -> dict[str, Any]:
    model = _get_bge_m3()
    return model.encode(
        texts,
        batch_size=max(1, Config.EMBEDDING_BATCH_SIZE),
        max_length=Config.BGE_M3_MAX_LENGTH,
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=False,
    )


def _lexical_weights_to_sparse(lexical_weights: dict[Any, Any]) -> qmodels.SparseVector:
    indices = [int(token_id) for token_id in lexical_weights]
    values = [float(lexical_weights[token_id]) for token_id in lexical_weights]
    return qmodels.SparseVector(indices=indices, values=values)


def dense_dimension() -> int:
    global _dense_dim
    if _dense_dim is None:
        output = _encode_bge_m3(["probe"], return_dense=True, return_sparse=False)
        _dense_dim = len(output["dense_vecs"][0])
        log.info("BGE-M3 dense dimension: %s", _dense_dim)
    return _dense_dim


def embed_chunks(chunks: list[str]) -> tuple[list[list[float]], list[qmodels.SparseVector]]:
    """Return BGE-M3 dense and learned sparse vectors for Qdrant upsert."""
    if not chunks:
        return [], []

    batch_size = max(1, Config.EMBEDDING_BATCH_SIZE)
    total = len(chunks)
    dense_vectors: list[list[float]] = []
    sparse_vectors: list[qmodels.SparseVector] = []

    log.info("Embedding %s chunk(s) with BGE-M3 in batches of %s...", total, batch_size)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        output = _encode_bge_m3(batch, return_dense=True, return_sparse=True)
        dense_vectors.extend(output["dense_vecs"])
        for lexical in output["lexical_weights"]:
            sparse_vectors.append(_lexical_weights_to_sparse(lexical))

    log.info("Finished embedding %s chunk(s).", total)
    return dense_vectors, sparse_vectors


def embed_query_sparse(text: str) -> qmodels.SparseVector:
    output = _encode_bge_m3([text], return_dense=True, return_sparse=True)
    return _lexical_weights_to_sparse(output["lexical_weights"][0])


class BGEM3Embeddings:
    """Dense embed_query / embed_documents interface used by the retriever and RAPTOR."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output = _encode_bge_m3(texts, return_dense=True, return_sparse=False)
        return output["dense_vecs"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> BGEM3Embeddings:
    _get_bge_m3()
    return BGEM3Embeddings()
