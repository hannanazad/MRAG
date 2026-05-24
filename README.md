# MRAG — MUTCD Multimodal RAG

A multimodal retrieval-augmented generation system over the **Manual on
Uniform Traffic Control Devices (MUTCD), 11th Edition**, designed to run
on TAMU HPRC (A100 / H100 GPUs).

## Architecture (v3)

```
PDF ─▶ outline-driven typed-paragraph chunks (Standard / Guidance / Option / Support)
   ─▶ caption-anchored figure / table crops
   ─▶ sign-code dictionary (R1-1 → "STOP sign", …) by category (Regulatory / Warning / …)
   ─▶ NetworkX knowledge graph
        ◦ Parts / Chapters / Sections / Chunks / Figures / SignCodes / Categories
        ◦ contains, cites_*, defines, depicts, mentions, illustrated_by, kind_of
   ─▶ Qdrant local-file store
        ◦ chunks   (BGE-M3 dense + sparse)
        ◦ figures  (BGE-M3 dense)
        ◦ pages    (ColQwen2-v0.1 multi-vector, binary-quantized)

query
  ─▶ hybrid retrieval (BGE-M3 dense + sparse, RRF)
  ─▶ scoring = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·rule_type
  ─▶ mxbai-rerank-large-v2
  ─▶ figures via graph cross-links (+ caption retrieval fallback)
  ─▶ pages via ColPali MAX_SIM late-interaction
  ─▶ Qwen2.5-VL-7B-Instruct (3B fallback) with rule-type-structured prompt
  ─▶ structured answer: Standards / Guidance / Options / Visual evidence / Citations
```

Detailed design: [`docs/architecture.md`](docs/architecture.md).

## Quick start (TAMU HPRC)

1. Read [`HPRC_SETUP.md`](HPRC_SETUP.md) once.
2. Put `mutcd11theditionr1hl.pdf` (or any `*.pdf`) in `$SCRATCH/MRAG/`.
3. Create the `mrag` conda env in `$SCRATCH/envs/mrag`, install `requirements.txt`.
4. Pre-cache the four checkpoints (BGE-M3, ColQwen2, mxbai-rerank-v2,
   Qwen2.5-VL-7B+3B) into `$HF_HOME`.
5. `sbatch scripts/ingest_v3.slurm`  (one-time, ~30–60 min on A100).
6. Open `MUTCD_MRAG_HPRC.ipynb` in OnDemand JupyterLab (A100 GPU, kernel
   `Python (mrag)`, modules `Anaconda3 WebProxy`), **Run All**.
7. `ask("...")` in any cell.

## Repo contents

| File / folder                | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `Copy_of_MRAG.ipynb`         | Original Colab notebook, preserved unmodified                             |
| `MUTCD_MRAG_HPRC.ipynb`      | The v3 notebook — initialises pipeline, `ask()` UI, KG inspector         |
| `mrag/`                      | The Python package (parsing, KG, embeddings, retrieval, VLM, ask)         |
| `scripts/extract_figures.py` | Standalone figure extractor (kept from v2 for offline use)               |
| `scripts/ingest_v3.py`       | One-shot ingestion driver                                                 |
| `scripts/ingest_v3.slurm`    | SLURM wrapper for the above                                              |
| `requirements.txt`           | Pinned deps for the `mrag` conda env                                      |
| `HPRC_SETUP.md`              | Step-by-step setup walkthrough                                            |
| `docs/architecture.md`       | Full design, schema, scoring formula, justifications                      |
| `README.md`                  | This file                                                                |

## Module layout (`mrag/`)

```
mrag/
  __init__.py          - package version
  config.py            - all paths, model names, retrieval / scoring weights
  parsing.py           - PDF outline → typed-paragraph chunks
  figures.py           - caption-anchored figure / table cropping + page render
  sign_codes.py        - sign-code regex + canonical name mining + categorisation
  kg.py                - NetworkX MultiDiGraph build + query API
  embeddings.py        - BGE-M3 (text), ColQwen2 (image), mxbai-rerank wrappers
  vector_store.py      - Qdrant local-file wrapper (three collections)
  retrieval.py         - hybrid + graph expansion + scoring + rerank
  vlm.py               - Qwen2.5-VL-7B loader + structured prompt + generation
  ask.py               - public `ask()` façade, inline display
```
