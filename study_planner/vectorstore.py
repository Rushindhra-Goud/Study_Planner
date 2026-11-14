"""A small persistent vector store (numpy matrix + JSONL chunks + JSON metadata)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import EmbeddingError, make_embedder
from .models import Chunk, TopicInfo


class VectorStoreError(RuntimeError):
    pass


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, chunks: List[Chunk], matrix: np.ndarray, embedder, meta: Dict[str, Any]):
        if len(chunks) != matrix.shape[0]:
            raise VectorStoreError("chunk count does not match embedding matrix")
        self.chunks = chunks
        self.matrix = matrix
        self.embedder = embedder
        self.meta = meta
        self.order: Dict[str, int] = {c.chunk_id: i for i, c in enumerate(chunks)}

    def __repr__(self) -> str:
        return f"VectorStore(chunks={len(self.chunks)}, backend={self.backend!r})"

    @property
    def backend(self) -> str:
        return str(self.meta.get("backend", "tfidf"))

    # ------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, chunks: List[Chunk], backend: str = "tfidf", model: Optional[str] = None,
              notes: Optional[Dict[str, Any]] = None) -> "VectorStore":
        if not chunks:
            raise VectorStoreError("no chunks to index")
        embedder = make_embedder(backend, model)
        texts = [c.embed_text() for c in chunks]
        embedder.fit(texts)
        matrix = embedder.embed_documents(texts)
        meta = {
            "backend": embedder.backend,
            "model": embedder.model,
            "dim": int(matrix.shape[1]),
            "n_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(notes or {}),
        }
        return cls(chunks, matrix, embedder, meta)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.matrix)
        arrays = self.embedder.arrays()
        if arrays:
            np.savez(directory / "embedder_state.npz", **arrays)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def exists(cls, directory: Path) -> bool:
        directory = Path(directory)
        return all((directory / name).exists() for name in ("chunks.jsonl", "embeddings.npy", "meta.json"))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        if not cls.exists(directory):
            raise VectorStoreError(
                f"No vector store found in {directory}. Run:  python -m study_planner ingest"
            )
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        chunks = []
        with open(directory / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))
        matrix = np.load(directory / "embeddings.npy")
        idf = None
        state = directory / "embedder_state.npz"
        if state.exists():
            with np.load(state) as data:
                idf = data["idf"]
        try:
            embedder = make_embedder(meta.get("backend", "tfidf"), meta.get("model"), idf=idf,
                                     dim=int(meta.get("dim", 4096)))
        except EmbeddingError as exc:
            raise VectorStoreError(str(exc)) from exc
        return cls(chunks, matrix, embedder, meta)

    # ------------------------------------------------------------ querying
    def search(self, query: str, k: int = 6, topic: Optional[str] = None,
               source: Optional[str] = None) -> List[Hit]:
        if not query.strip():
            return []
        mask = np.ones(len(self.chunks), dtype=bool)
        if topic is not None:
            mask &= np.array([c.topic == topic for c in self.chunks])
        if source is not None:
            mask &= np.array([c.source == source for c in self.chunks])
        if not mask.any():
            return []
        q = self.embedder.embed_query(query)
        scores = self.matrix @ q
        scores = np.where(mask, scores, -np.inf)
        k = max(1, min(k, int(mask.sum())))
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def chunks_for_topic(self, topic: str) -> List[Chunk]:
        return [c for c in self.chunks if c.topic == topic]

    def topics(self) -> List[TopicInfo]:
        info: Dict[str, TopicInfo] = {}
        for c in self.chunks:
            if c.topic not in info:
                info[c.topic] = TopicInfo(c.topic, c.doc_title, c.source, 0, 0)
            info[c.topic].n_chunks += 1
            info[c.topic].n_chars += len(c.text)
        return list(info.values())
