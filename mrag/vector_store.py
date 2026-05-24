"""Qdrant local-file vector store: one DB folder, three collections.

Collections:
  mutcd_chunks   - chunks: dense + sparse, rich payload
  mutcd_figures  - figures: dense on caption+title, payload incl. image_path
  mutcd_pages    - pages:   ColPali multi-vector with binary quantization

All run in *embedded* mode via `QdrantClient(path=...)`. No daemon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mrag.vector_store")


@dataclass
class ChunkRow:
    id:       int                 # stable integer id (hash of chunk_id)
    dense:    np.ndarray
    sparse:   Dict[int, float]
    payload:  Dict[str, Any]


@dataclass
class FigureRow:
    id:       int
    dense:    np.ndarray
    payload:  Dict[str, Any]


@dataclass
class PageRow:
    id:       int
    vectors:  np.ndarray          # (num_patches, dim)
    payload:  Dict[str, Any]


class VectorStore:
    def __init__(self, qdrant_dir: Path) -> None:
        self.qdrant_dir = Path(qdrant_dir)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)
        from qdrant_client import QdrantClient
        self._client = QdrantClient(path=str(self.qdrant_dir))

    @property
    def client(self):
        return self._client

    # ----- schema -----------------------------------------------------------

    def init_collections(
        self,
        coll_chunks: str,
        coll_figures: str,
        coll_pages: str,
        text_dim: int = 1024,
        page_patch_dim: int = 128,
        use_binary_quantization_for_pages: bool = True,
    ) -> None:
        from qdrant_client.http import models as qm

        def recreate(name: str, vectors_config, sparse_vectors_config=None,
                     quantization_config=None):
            self._client.recreate_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config or {},
                quantization_config=quantization_config,
            )

        # 1. Chunks: named dense vector + named sparse vector
        recreate(
            coll_chunks,
            vectors_config={
                "dense": qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(),
            },
        )
        # 2. Figures: dense only (caption + sign codes embedded)
        recreate(
            coll_figures,
            vectors_config={
                "dense": qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            },
        )
        # 3. Pages: ColPali multi-vector with optional binary quantization
        qcfg = (
            qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True))
            if use_binary_quantization_for_pages else None
        )
        recreate(
            coll_pages,
            vectors_config={
                "colbert": qm.VectorParams(
                    size=page_patch_dim,
                    distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(
                        comparator=qm.MultiVectorComparator.MAX_SIM,
                    ),
                    quantization_config=qcfg,
                ),
            },
        )

    # ----- ingestion --------------------------------------------------------

    def upsert_chunks(self, name: str, rows: List[ChunkRow], batch: int = 256) -> None:
        from qdrant_client.http import models as qm
        for i in range(0, len(rows), batch):
            slice_ = rows[i:i+batch]
            self._client.upsert(
                collection_name=name,
                points=[
                    qm.PointStruct(
                        id=r.id,
                        vector={
                            "dense": r.dense.tolist(),
                            "sparse": qm.SparseVector(
                                indices=list(r.sparse.keys()),
                                values=list(r.sparse.values()),
                            ),
                        },
                        payload=r.payload,
                    )
                    for r in slice_
                ],
                wait=True,
            )

    def upsert_figures(self, name: str, rows: List[FigureRow], batch: int = 256) -> None:
        from qdrant_client.http import models as qm
        for i in range(0, len(rows), batch):
            slice_ = rows[i:i+batch]
            self._client.upsert(
                collection_name=name,
                points=[
                    qm.PointStruct(
                        id=r.id,
                        vector={"dense": r.dense.tolist()},
                        payload=r.payload,
                    )
                    for r in slice_
                ],
                wait=True,
            )

    def upsert_pages(self, name: str, rows: List[PageRow], batch: int = 32) -> None:
        from qdrant_client.http import models as qm
        for i in range(0, len(rows), batch):
            slice_ = rows[i:i+batch]
            self._client.upsert(
                collection_name=name,
                points=[
                    qm.PointStruct(
                        id=r.id,
                        vector={"colbert": r.vectors.tolist()},
                        payload=r.payload,
                    )
                    for r in slice_
                ],
                wait=True,
            )

    # ----- search -----------------------------------------------------------

    def search_chunks_hybrid(
        self,
        name: str,
        dense: np.ndarray,
        sparse: Dict[int, float],
        top_k: int = 30,
    ):
        """Returns merged top-k via Reciprocal Rank Fusion of dense+sparse."""
        from qdrant_client.http import models as qm
        # Two parallel searches, then RRF.
        dense_hits = self._client.search(
            collection_name=name,
            query_vector=("dense", dense.tolist()),
            limit=top_k,
            with_payload=True,
        )
        sparse_hits = (
            self._client.search(
                collection_name=name,
                query_vector=qm.NamedSparseVector(
                    name="sparse",
                    vector=qm.SparseVector(
                        indices=list(sparse.keys()) or [0],
                        values=list(sparse.values()) or [0.0],
                    ),
                ),
                limit=top_k,
                with_payload=True,
            )
            if sparse else []
        )
        return _rrf_merge(dense_hits, sparse_hits, top_k=top_k)

    def search_figures(
        self,
        name: str,
        dense: np.ndarray,
        top_k: int = 8,
    ):
        return self._client.search(
            collection_name=name,
            query_vector=("dense", dense.tolist()),
            limit=top_k,
            with_payload=True,
        )

    def search_pages(
        self,
        name: str,
        multivec_query: np.ndarray,
        top_k: int = 6,
    ):
        from qdrant_client.http import models as qm
        return self._client.search(
            collection_name=name,
            query_vector=qm.NamedVector(name="colbert", vector=multivec_query.tolist()),
            limit=top_k,
            with_payload=True,
        )


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _rrf_merge(*result_lists, top_k: int = 30, k_rrf: int = 60) -> List[Any]:
    """Reciprocal Rank Fusion of multiple ScoredPoint lists."""
    scores: Dict[int, float] = {}
    payloads: Dict[int, Any] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            pid = hit.id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
            if pid not in payloads:
                payloads[pid] = hit
    ordered = sorted(payloads.items(), key=lambda kv: scores[kv[0]], reverse=True)
    out = []
    for pid, hit in ordered[:top_k]:
        # Attach fused score onto the hit for downstream
        hit_dict = {"id": pid, "score": scores[pid], "payload": hit.payload}
        out.append(hit_dict)
    return out


def chunk_id_to_int(chunk_id: str) -> int:
    """Stable positive 63-bit integer id from a chunk_id string."""
    import hashlib
    h = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False) >> 1
