#!/bin/bash
# ===========================================================================
# ingest.slurm — headless MUTCD ingestion on TAMU HPRC
# Renders page PNGs from the PDF, builds page records, and computes
# section + page embeddings. Saves everything under $SCRATCH/MRAG/mmrag_cache.
#
# Submit with:
#   sbatch scripts/ingest.slurm
#
# Adjust --partition, --account, --gres if your cluster uses different names.
# ===========================================================================

#SBATCH --job-name=mrag-ingest
#SBATCH --output=logs/ingest-%j.out
#SBATCH --error=logs/ingest-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

echo "[$(date)] node=$HOSTNAME job=$SLURM_JOB_ID"

# --- Modules + env ---------------------------------------------------------
module purge
module load Anaconda3
module load WebProxy      # needed if HF cache isn't fully pre-populated

# Keep all caches in scratch (HOME is too small)
export CONDA_ENVS_PATH=$SCRATCH/envs
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

source activate $SCRATCH/envs/mrag

# --- Run ingestion ---------------------------------------------------------
cd $SCRATCH/MRAG

python - <<'PY'
import os, json, re
from pathlib import Path
import numpy as np
import fitz                       # PyMuPDF
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

SCRATCH = Path(os.environ["SCRATCH"])
BASE_DIR = SCRATCH / "MRAG"
PDF_PATH = BASE_DIR / "mutcd11theditionr1hl.pdf"
SECTIONS_JSON = BASE_DIR / "mutcd_sections_with_images.json"
PAGE_IMAGE_DIR = BASE_DIR / "page_images"
CACHE_DIR = BASE_DIR / "mmrag_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RENDER_DPI = 200
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 1. render page images if missing
PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
doc = fitz.open(str(PDF_PATH))
print(f"PDF pages: {doc.page_count}")
zoom = RENDER_DPI / 72.0
mat = fitz.Matrix(zoom, zoom)
rendered = 0
for i, page in enumerate(tqdm(doc, desc="render pages")):
    out = PAGE_IMAGE_DIR / f"page_{i+1:04d}.png"
    if out.exists():
        continue
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(out))
    rendered += 1
print(f"Rendered {rendered} new page PNGs.")

# 2. page records
def norm(t):  return re.sub(r"\s+", " ", t or "").strip()
fig_re   = re.compile(r"(Figure\s+[A-Za-z0-9.\-]+[^.\n]*)", re.IGNORECASE)
tbl_re   = re.compile(r"(Table\s+[A-Za-z0-9.\-]+[^.\n]*)",  re.IGNORECASE)
sign_re  = re.compile(r"\b([A-Z]{1,3}\d{1,2}[A-Za-z0-9\-]*)\b")

records = []
for i, page in enumerate(tqdm(doc, desc="page records")):
    txt = page.get_text("text") or ""
    figs = [norm(x) for x in fig_re.findall(txt)[:3]]
    tbls = [norm(x) for x in tbl_re.findall(txt)[:3]]
    signs = sorted(set(sign_re.findall(txt)))[:50]
    records.append({
        "page_num": i + 1,
        "image_path": str(PAGE_IMAGE_DIR / f"page_{i+1:04d}.png"),
        "page_text": txt,
        "page_text_norm": norm(txt),
        "figure_captions": figs,
        "table_titles": tbls,
        "sign_codes": signs,
    })

with open(CACHE_DIR / "page_records.json", "w") as f:
    json.dump(records, f, ensure_ascii=False)
print(f"page_records.json saved ({len(records)} pages)")

# 3. embeddings
with open(SECTIONS_JSON) as f:
    raw_sections = json.load(f)
section_texts = [
    f"Section {s.get('section','')}\nTitle: {s.get('title','')}\nPage: {s.get('page_num','')}\n{s.get('text','')}"
    for s in raw_sections
]
page_texts = [
    f"Page {p['page_num']}\nFigure captions: {' | '.join(p['figure_captions'])}\n"
    f"Table titles: {' | '.join(p['table_titles'])}\n"
    f"Sign codes: {' | '.join(p['sign_codes'][:20])}\n{p['page_text'][:4000]}"
    for p in records
]

embed = SentenceTransformer(EMBED_MODEL_NAME)
print("Encoding sections...")
sec_emb = embed.encode(section_texts, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True).astype("float32")
np.save(CACHE_DIR / "section_embeddings.npy", sec_emb)
print("Encoding pages...")
pg_emb = embed.encode(page_texts, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=True).astype("float32")
np.save(CACHE_DIR / "page_embeddings.npy", pg_emb)
print("DONE. Cache at:", CACHE_DIR)
PY

echo "[$(date)] ingestion complete"
