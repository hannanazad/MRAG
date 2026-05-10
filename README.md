# MRAG — MUTCD Multimodal RAG

Dual-retrieval (text sections + full pages) Multimodal RAG over the
**Manual on Uniform Traffic Control Devices (MUTCD)**, answered by
`Qwen/Qwen2.5-VL-3B-Instruct`.

## Repo contents

| File / folder                | Purpose                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| `Copy_of_MRAG.ipynb`         | Original Colab notebook (kept for reference; Drive-mounted; final UI cell is the working one) |
| `MUTCD_MRAG_HPRC.ipynb`      | **Use this on HPRC.** End-to-end pipeline adapted for TAMU HPRC JupyterLab                    |
| `HPRC_SETUP.md`              | Step-by-step setup for TAMU HPRC (modules, conda env, OnDemand, gradio access, troubleshooting) |
| `requirements.txt`           | Python deps for the `mrag` conda env                                                           |
| `scripts/ingest.slurm`       | Optional SLURM job that renders page PNGs + builds embeddings off the JupyterLab GPU          |

## Quick start

1. Read **`HPRC_SETUP.md`** end-to-end (one-time setup).
2. Put `mutcd11theditionr1hl.pdf` and `mutcd_sections_with_images.json` (and optionally `page_images/`) under `$SCRATCH/MRAG/`.
3. Create the `mrag` conda env in `$SCRATCH/envs/mrag` from `requirements.txt`.
4. Open OnDemand → **JupyterLab** with modules `Anaconda3 WebProxy`, conda env `mrag`, **1 GPU**.
5. Open `MUTCD_MRAG_HPRC.ipynb`, switch kernel to **Python (mrag)**, run all.
6. Use the OnDemand proxy URL the notebook prints (or the gradio.live link if `WebProxy` is loaded).

## What changed vs. the Colab notebook

- Storage `/content/drive/...` → `$SCRATCH/MRAG/...`
- HF + pip caches forced into `$SCRATCH/hf_cache` (HOME quota is small)
- Missing `page_images/` are auto-rendered from the PDF by the notebook
- VLM loader has explicit `Qwen2_5_VLForConditionalGeneration` path + pipeline fallback
- Broken alt pipeline (Colab cell 19 referencing undefined `section_chunks` and `llm`) removed; one consolidated pipeline
- Gradio UI now also prints the OnDemand reverse-proxy URL, not just `share=True`

See `HPRC_SETUP.md` for full detail.
