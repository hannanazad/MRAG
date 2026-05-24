"""Retrieval pipeline.

  query
   │
   ├─ parse explicit ids / sign codes ────► direct lookups
   ├─ BGE-M3 dense+sparse hybrid (RRF) ───► top-K1 chunks
   ├─ +graph 1-hop expansion              ► augment candidate set
   ├─ apply scoring formula:
   │     S = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·w(content_type)
   ├─ mxbai-rerank-large-v2 over top-K1 ──► top-K2 chunks
   ├─ ColQwen2 page retrieval (parallel) ─► top-K3 pages
   └─ pull cross-linked figures from winning chunks (via KG)
        + extra figure retrieval (BGE-M3 on caption/title)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import CFG
from .embeddings import TextEmbedder, ImageEmbedder, Reranker
from .kg import KG
from .vector_store import VectorStore

log = logging.getLogger("mrag.retrieval")


@dataclass
class RetrievalResult:
    chunks:  List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    pages:   List[Dict[str, Any]] = field(default_factory=list)
    debug:   Dict[str, Any]       = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        kg: KG,
        text_embedder: TextEmbedder,
        image_embedder: Optional[ImageEmbedder],
        reranker: Reranker,
    ) -> None:
        self.store = store
        self.kg = kg
        self.text = text_embedder
        self.img  = image_embedder
        self.rerank = reranker

    # ----- public entry point ----------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        result = RetrievalResult()
        result.debug["query"] = query

        # 1. Direct lookups -------------------------------------------------
        explicit = self.kg.query_entities(query)
        result.debug["query_entities"] = list(explicit)

        # 2. Hybrid chunk retrieval -----------------------------------------
        dense, sparse_list = self.text.encode_both([query])
        dense_q = dense[0]
        sparse_q = sparse_list[0]
        fused = self.store.search_chunks_hybrid(
            CFG.coll_chunks, dense_q, sparse_q, top_k=CFG.top_k_fused,
        )

        # 3. Graph expansion ------------------------------------------------
        candidate_ids: Set[int] = {h["id"] for h in fused}
        for ent_node in explicit:
            for nb in self.kg.neighbors(ent_node, n_hops=1):
                if not nb.startswith("chunk:"):
                    continue
                # Fetch this chunk by chunk_id from Qdrant via scroll (rare path).
                # In practice we just bump scoring for these in step 4.
                pass  # placeholder; scoring handles it.

        # 4. Apply the scoring formula --------------------------------------
        scored = []
        for hit in fused:
            payload = hit["payload"] or {}
            chunk_id = payload.get("chunk_id", "")
            base = float(hit.get("score", 0.0))
            s_graph = self.kg.proximity_score(explicit, chunk_id)
            s_rt = CFG.rule_type_weight(payload.get("content_type", "Support"))
            s_hier = _hierarchy_prior(query, payload)
            final = (
                CFG.w_dense   * base
                + CFG.w_graph * s_graph
                + CFG.w_ruletype * (s_rt - 1.0)            # center at 1.0
                + CFG.w_hierarchy * s_hier
            )
            scored.append((final, hit))
        scored.sort(key=lambda t: t[0], reverse=True)
        precursor = [h for _s, h in scored[: CFG.top_k_after_graph]]

        # 5. Cross-encoder rerank ------------------------------------------
        docs = [h["payload"].get("text", "")[:1500] for h in precursor]
        rerank_pairs = self.rerank.rank(query, docs, top_k=CFG.top_k_after_rerank)
        final_chunks = []
        for idx, score in rerank_pairs:
            hit = precursor[idx]
            payload = hit["payload"] or {}
            final_chunks.append({**payload, "score": score})
        result.chunks = final_chunks

        # 6. Figures by chunk cross-links + caption retrieval --------------
        figure_ids_seen: Set[str] = set()
        figs_out: List[Dict[str, Any]] = []
        for ch in final_chunks:
            for fid in self.kg.figures_for_chunk(ch.get("chunk_id", "")):
                if fid in figure_ids_seen: continue
                figure_ids_seen.add(fid)
                payload = _figure_payload_from_graph(self.kg, fid)
                if payload:
                    figs_out.append(payload)
        # Top up with caption retrieval if too few.
        if len(figs_out) < CFG.top_k_figures:
            extra_hits = self.store.search_figures(CFG.coll_figures, dense_q, top_k=CFG.top_k_figures)
            for h in extra_hits:
                payload = h.payload or {}
                fid = payload.get("figure_id")
                if fid and fid not in figure_ids_seen:
                    figure_ids_seen.add(fid)
                    figs_out.append({**payload, "score": float(h.score)})
                if len(figs_out) >= CFG.top_k_figures:
                    break
        result.figures = figs_out

        # 7. ColPali page retrieval (optional) ------------------------------
        if self.img is not None:
            try:
                q_mv = self.img.encode_queries([query])[0]
                page_hits = self.store.search_pages(CFG.coll_pages, q_mv, top_k=CFG.top_k_pages)
                result.pages = [
                    {**(h.payload or {}), "score": float(h.score)}
                    for h in page_hits
                ]
            except Exception as e:
                log.warning("ColPali page retrieval failed: %r", e)

        return result


def _hierarchy_prior(query: str, payload: Dict[str, Any]) -> float:
    """Cheap prior on top of dense+sparse: if the query mentions Part N or
    Chapter NX, give chunks in that branch a small boost."""
    score = 0.0
    q = query.lower()
    part = (payload.get("part") or "").lower()
    chapter = (payload.get("chapter") or "").lower()
    m_part = re.search(r"\bpart\s+(\d+)\b", q)
    if m_part and f"part {m_part.group(1)}" in part:
        score += 0.6
    m_chap = re.search(r"\bchapter\s+([0-9a-z]+)\b", q)
    if m_chap and m_chap.group(1) in chapter:
        score += 0.6
    return score


def _figure_payload_from_graph(kg: KG, figure_id: str) -> Optional[Dict[str, Any]]:
    node = kg.figure(figure_id)
    if not node:
        return None
    data = kg.g.nodes[node]
    return {
        "figure_id":     data.get("id", figure_id),
        "page_pdf":      data.get("page_pdf"),
        "page_printed":  data.get("page_printed"),
        "caption":       data.get("caption", ""),
        "image_path":    data.get("image_path", ""),
        "sign_codes":    list(data.get("sign_codes", [])),
        "source":        "graph_link",
    }
