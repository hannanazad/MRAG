"""
MRAG Central Configuration
===========================
All storage paths and model settings live here.
To move your data to a different location, change STORAGE_ROOT only.
To swap the VLM, change VLM_PROVIDER and VLM_MODEL only.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# STORAGE — change this one line to move everything
# ─────────────────────────────────────────────────────────────────────────────

# Options (uncomment whichever applies):
#   Google Drive (Colab):  "/content/drive/MyDrive/MRAG"
#   HPRC $SCRATCH:         os.environ.get("SCRATCH", "/scratch") + "/MRAG"
#   Local:                 str(Path.home() / "MRAG")

STORAGE_ROOT = "/content/drive/MyDrive/MRAG"   # ← CHANGE ONLY THIS LINE

# All sub-paths are derived automatically — do not edit below this line
STORAGE_ROOT   = Path(STORAGE_ROOT)
PDF_PATH       = STORAGE_ROOT / "mutcd.pdf"         # source PDF
PAGE_PNG_DIR   = STORAGE_ROOT / "pages"             # rendered page images
FIGURE_DIR     = STORAGE_ROOT / "figures"           # cropped figure images
QDRANT_PATH    = STORAGE_ROOT / "qdrant_db"         # vector store
GRAPH_PATH     = STORAGE_ROOT / "graph.gpickle"     # NetworkX knowledge graph
SIGN_CODES_PATH= STORAGE_ROOT / "sign_codes.json"   # sign code dictionary
FIGURES_JSONL  = STORAGE_ROOT / "figures.jsonl"     # figure metadata

# HuggingFace model cache — redirected away from ~/.cache to avoid filling
# Google Drive or home quota. Set to a fast local disk.
HF_CACHE_DIR   = STORAGE_ROOT / "hf_cache"

# Apply HF cache redirect immediately on import so it takes effect before
# any transformers/huggingface_hub imports elsewhere.
os.environ["HF_HOME"]             = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# VLM (Visual Language Model) — change these two lines to swap models
# ─────────────────────────────────────────────────────────────────────────────

# VLM_PROVIDER options:
#   "api"   → use an OpenAI-compatible REST API (no GPU / no local download)
#   "local" → load model weights onto local GPU via HuggingFace transformers

VLM_PROVIDER = "api"       # ← "api" or "local"

# When VLM_PROVIDER == "api":
#   Model name string sent to the API endpoint.
#   Examples:
#     Qwen3 VL 32B (current):   "qwen3-vl-32b-instruct"
#     Qwen2.5 VL 72B:           "qwen2.5-vl-72b-instruct"
#     GPT-4o:                   "gpt-4o"              (change API_BASE_URL too)
#     Claude claude-opus-4-6:      "claude-opus-4-6"       (use Anthropic SDK instead)
#
# When VLM_PROVIDER == "local":
#   HuggingFace model ID downloaded to HF_CACHE_DIR.
#   Examples:
#     "Qwen/Qwen2.5-VL-7B-Instruct"   (original, ~17 GB)
#     "Qwen/Qwen2.5-VL-3B-Instruct"   (lighter fallback, ~8 GB)

VLM_MODEL = "qwen3-vl-32b-instruct"   # ← CHANGE THIS to swap models

# API endpoint — only used when VLM_PROVIDER == "api"
# Qwen / DashScope OpenAI-compatible endpoint:
API_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# For OpenRouter (access many models through one key):
# API_BASE_URL = "https://openrouter.ai/api/v1"
# For standard OpenAI (GPT-4o etc.):
# API_BASE_URL = "https://api.openai.com/v1"

# API key — read from environment variable so it is never hard-coded in source.
# Set it in your Colab / HPRC session with:
#   import os; os.environ["VLM_API_KEY"] = "sk-..."
# Or add it to your ~/.bashrc / job script:
#   export VLM_API_KEY="sk-..."
API_KEY_ENV_VAR = "VLM_API_KEY"   # name of the env var that holds your key

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval / generation parameters — safe to leave as-is
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_CHUNKS   = 30    # candidates from hybrid vector search
RERANK_TOP_K   = 6     # chunks kept after cross-encoder rerank
TOP_K_PAGES    = 4     # page images sent to VLM
MAX_NEW_TOKENS = 1024  # max tokens in VLM answer

# Scoring weights  S = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·rule
ALPHA   = 1.0   # dense vector similarity
BETA    = 0.6   # sparse (BM25-style) similarity
GAMMA   = 0.2   # document hierarchy proximity
DELTA   = 0.4   # knowledge-graph proximity
EPSILON = 0.3   # rule-type weight (Standard > Guidance > Option > Support)
