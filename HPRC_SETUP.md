# Running the MUTCD Multimodal RAG on TAMU HPRC

This guide walks you through migrating the Colab notebook
(`Copy_of_MRAG.ipynb`) to the HPRC-ready notebook
(`MUTCD_MRAG_HPRC.ipynb`) and running it on **Grace**, **FASTER**,
**Launch**, or **ACES** via the **Open OnDemand JupyterLab** portal.

The HPRC-side differences from Colab in one line:
**no Drive mount → use `$SCRATCH`, request a GPU through OnDemand, load
the `WebProxy` module so the compute node can reach the internet, and
point Hugging Face caches at `$SCRATCH` so you don’t blow out your HOME
quota.**

---

## 1. What you need on HPRC before you start

1. An active HPRC account on one of the clusters (Grace / FASTER /
   Launch / ACES). Apply at <https://hprc.tamu.edu/apply/>.
2. Your NetID and Duo MFA configured.
3. The three input artifacts from your Colab project:
   - `mutcd11theditionr1hl.pdf`
   - `mutcd_sections_with_images.json`
   - `page_images/` (the pre-rendered per-page PNGs, `page_0001.png`, …)

   If you only have the PDF, the notebook can regenerate
   `page_images/` for you (see Step 7).

You will put everything under `$SCRATCH/MRAG/`. **Never** put the PDF or
images in `$HOME` — HOME is tiny (~10 GB) and scratch is ~1 TB.

---

## 2. Pick a cluster

| Cluster | GPU options                          | Good for                              |
| ------- | ------------------------------------ | ------------------------------------- |
| FASTER  | T4 (16 GB), A100 (40 GB) “composable” | Best general choice for Qwen2.5-VL-3B |
| Launch  | H100 (80 GB), A30 (24 GB)            | Fastest; newest stack                 |
| ACES    | H100 / A100 / A30 / PVC / Gaudi      | Lots of accelerator variety           |
| Grace   | A100 (40/80 GB), RTX 6000 (24 GB), T4 | Solid fallback                        |

Qwen2.5-VL-3B-Instruct in bf16 needs **~7–8 GB VRAM**, so any GPU above
≥ 12 GB is comfortable (T4, A30, A100, H100, RTX 6000 all work).

Portal URLs (log in with NetID + Duo):

- Grace:  <https://portal-grace.hprc.tamu.edu>
- FASTER: <https://portal-faster.hprc.tamu.edu>
- Launch: <https://portal-launch.hprc.tamu.edu>
- ACES:   <https://portal-aces.hprc.tamu.edu>

---

## 3. Upload your data to `$SCRATCH`

You have three easy options. Use whichever is convenient.

### 3a. Open OnDemand → Files (browser, easiest)

1. Log into the cluster’s portal above.
2. Top bar → **Files → /scratch/user/<NetID>**.
3. Create a folder `MRAG`.
4. Click **Upload** and drag in your PDF and JSON. For `page_images/`,
   zip it first (`page_images.zip`) and upload the zip — much faster than
   thousands of small files.
5. After upload, open a portal **Shell Access** and unzip:

   ```bash
   cd $SCRATCH/MRAG
   unzip -q page_images.zip
   ```

### 3b. From your laptop with `scp` / `rsync` (fastest for big folders)

Use the **data transfer node**, not a login node:

```bash
# Grace
rsync -avh ./MRAG/  <NetID>@grace-dtn1.hprc.tamu.edu:/scratch/user/<NetID>/MRAG/
# FASTER
rsync -avh ./MRAG/  <NetID>@faster-dtn1.hprc.tamu.edu:/scratch/user/<NetID>/MRAG/
# Launch
rsync -avh ./MRAG/  <NetID>@launch-dtn1.hprc.tamu.edu:/scratch/user/<NetID>/MRAG/
```

### 3c. Pull from your Google Drive

In an OnDemand shell on a login node (login nodes have outbound
internet without WebProxy):

```bash
module load WebProxy   # optional on login nodes but harmless
pip install --user gdown
cd $SCRATCH/MRAG
gdown --folder "<your Drive folder share link>"
```

Verify the layout:

```bash
ls $SCRATCH/MRAG
# expected:
#   mutcd11theditionr1hl.pdf
#   mutcd_sections_with_images.json
#   page_images/          (page_0001.png, page_0002.png, ...)
```

---

## 4. Clone this repo into `$SCRATCH`

In an OnDemand **Shell Access** session (or `ssh <NetID>@grace.hprc.tamu.edu`):

```bash
cd $SCRATCH
git clone https://github.com/hannanazad/MRAG.git
cd MRAG
ls
# Copy_of_MRAG.ipynb     <- original Colab notebook (left intact)
# MUTCD_MRAG_HPRC.ipynb  <- HPRC-ready notebook (use this one)
# requirements.txt
# HPRC_SETUP.md          <- this file
# scripts/ingest.slurm   <- optional batch ingestion job
```

---

## 5. Create a conda env in `$SCRATCH` (do this once)

HPRC HOME has a small quota — **never** install conda envs there.
Put them in `$SCRATCH/envs`.

```bash
# On a LOGIN node (login nodes have internet without WebProxy)
module purge
module load Anaconda3/2024.02-1            # exact name varies per cluster; `module avail Anaconda3`
module load WebProxy                       # safe to load; needed if you ever do this on a compute node

# Tell conda + pip + HF to keep everything in $SCRATCH
export CONDA_ENVS_PATH=$SCRATCH/envs
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

conda create -y -p $SCRATCH/envs/mrag python=3.11
# On HPRC, use `source activate` NOT `conda activate`.
# The Lmod-loaded Anaconda module does not run `conda init`, so
# `conda activate` will error with "Run 'conda init' before 'conda activate'".
# `source activate` works without conda init.
source activate $SCRATCH/envs/mrag

# Install torch matching the cluster CUDA module you'll use later.
# Most HPRC GPU images ship CUDA 12.x; cu121 wheels work everywhere I’ve tested.
pip install --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.4.1 torchvision==0.19.1

# Then everything else
pip install -r $SCRATCH/MRAG/requirements.txt

# Register the kernel so JupyterLab can pick it.
python -m ipykernel install --user --name mrag --display-name "Python (mrag)"
```

If `Anaconda3/2024.02-1` is not in `module avail`, just run
`module avail Anaconda3` and pick the newest version listed.

> Tip: put the env-variable exports into a file
> `$SCRATCH/MRAG/env.sh` and `source` it whenever you start a new shell:
>
> ```bash
> cat > $SCRATCH/MRAG/env.sh <<'EOF'
> export CONDA_ENVS_PATH=$SCRATCH/envs
> export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
> export PIP_CACHE_DIR=$SCRATCH/pip_cache
> export HF_HOME=$SCRATCH/hf_cache
> export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
> module load Anaconda3
> module load WebProxy
> source activate $SCRATCH/envs/mrag
> EOF
> chmod +x $SCRATCH/MRAG/env.sh
> ```

---

## 6. Pre-download the model on the login node (recommended)

Compute nodes are fastest when models are already cached. Run this once
on a login node so the ~8 GB Qwen download lives in `$SCRATCH/hf_cache`:

```bash
source $SCRATCH/MRAG/env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print("Models cached under", __import__("os").environ.get("HF_HOME"))
PY
```

If this is the **first time** the model is used you may need to
accept the model’s license on Hugging Face once
(<https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>) and run
`huggingface-cli login` with an HF token (read-scope is fine).

---

## 7. Launch JupyterLab through OnDemand

1. Portal → **Interactive Apps → JupyterLab**.
2. Fill in:
   - **Modules to load**: `Anaconda3 WebProxy`
     (the `WebProxy` module is what gives the compute node outbound
     network access — without it the gradio.live tunnel and any HF
     download from inside the notebook will hang).
   - **Use Conda Environment**: tick it and point to
     `/scratch/user/<NetID>/envs/mrag`.
   - **Number of cores**: 4–8.
   - **Memory**: 32 GB is plenty.
   - **GPUs**: 1 (T4 / A30 / A100 / H100 — anything ≥ 12 GB).
   - **Walltime**: 2–4 hours for development.
3. Launch, wait for **Connect to JupyterLab**.
4. In the file tree open `MRAG/MUTCD_MRAG_HPRC.ipynb`.
5. Top right kernel switcher → **Python (mrag)**.
6. Run cells top to bottom.

---

## 8. Differences from the Colab notebook (already wired up for you)

`MUTCD_MRAG_HPRC.ipynb` is the Colab notebook adapted for HPRC. Concretely:

| Concern             | Colab                                  | HPRC notebook                                                                              |
| ------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------ |
| Storage             | `/content/drive/MyDrive/MRAG`          | `$SCRATCH/MRAG` (read from `os.environ["SCRATCH"]`)                                        |
| Mounting            | `drive.mount(...)`                     | none                                                                                       |
| Package install     | `pip install ...` in the notebook      | done once, ahead of time, into the `mrag` conda env (notebook just `import`s)              |
| HF cache            | ephemeral in `/root/.cache`            | `$SCRATCH/hf_cache` (env vars set in cell 0)                                               |
| GPU detection       | implicit                               | explicit `torch.cuda.is_available()` print, plus a `device_map="auto"` + `torch_dtype=bf16` |
| Page images missing | falls back silently                    | offers to render them from the PDF with PyMuPDF inline (Step 7 below)                      |
| Broken `llm` cell   | cell 19 references undefined `llm`     | removed; pipeline is consolidated into one working flow                                    |
| Gradio sharing      | `share=True` only                      | tries OnDemand proxy URL first, falls back to `share=True` (needs `WebProxy`)              |

If `page_images/` is missing on HPRC, run the rendering cell in the
notebook — it uses PyMuPDF to dump `page_XXXX.png` at 200 DPI in a few
minutes. Or run the SLURM job below to do it without occupying your
JupyterLab GPU session.

---

## 9. (Optional) Run ingestion as a SLURM batch job

If you don’t want to spend GPU walltime rendering pages and building
embeddings inside JupyterLab, submit the included batch script. It will
render `page_images/`, build page records, and save the embeddings
under `$SCRATCH/MRAG/mmrag_cache/` so the JupyterLab session is
purely interactive.

```bash
cd $SCRATCH/MRAG
sbatch scripts/ingest.slurm
squeue -u $USER          # watch it
tail -f logs/ingest-*.out
```

When it finishes, `mmrag_cache/section_embeddings.npy`,
`mmrag_cache/page_embeddings.npy`, and `mmrag_cache/page_records.json`
will exist, and the notebook will load them instantly.

---

## 10. Gradio access on HPRC — read this once

Inside JupyterLab the gradio app is launched on
`server_name="0.0.0.0"`, default port 7860. Two ways to reach it:

1. **Easy & private — OnDemand proxy.** OnDemand exposes any port on
   the compute node at:

   ```
   https://portal-<cluster>.hprc.tamu.edu/rnode/<hostname>/<port>/
   ```

   The HPRC notebook prints this URL for you. Click it.

2. **Public tunnel — `share=True`.** Requires the `WebProxy` module
   loaded for the JupyterLab job (Step 7). The notebook will also print
   the `gradio.live` URL when this works.

If neither works the most common cause is forgetting the `WebProxy`
module in the JupyterLab launch form. Re-launch with it loaded.

---

## 11. Troubleshooting cheatsheet

- **Gradio launch fails with `... localhost:<port>/gradio_api/startup-events
  failed (code 503) ...`** — the `WebProxy` module set `http_proxy` and
  `https_proxy`, and Gradio's localhost handshake is being routed through
  the proxy. Fix:
  ```python
  import os
  os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,::1"
  os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
  ```
  Run that before `demo.launch(...)`. The HPRC notebook now does this
  automatically.
- **`CondaError: Run 'conda init' before 'conda activate'`** — use
  `source activate $SCRATCH/envs/mrag` instead of `conda activate ...`.
  The Lmod Anaconda module on HPRC does not run `conda init`. (Already
  fixed in the commands above; this is here for grep-ability.)
- **`OSError: ... No space left on device` during model download** — HF
  cache is going to `$HOME`. Re-run `source $SCRATCH/MRAG/env.sh` so
  `HF_HOME` points into scratch, then re-run the cell.
- **`CUDA out of memory`** — pick a bigger GPU in the OnDemand form, or
  set `VLM_DTYPE = torch.float16` (notebook has a switch), or lower
  `FINAL_PAGE_RESULTS` to 2.
- **Gradio hangs on launch, never prints a URL** — `WebProxy` not
  loaded. Cancel the kernel, restart the JupyterLab job with
  `WebProxy` in the modules field.
- **`fitz` import error** — your kernel is not the `mrag` kernel.
  Kernel menu → **Change Kernel → Python (mrag)**.
- **Embeddings cell takes forever** — that’s expected on the first
  run for `page_embeddings` (~1000 pages). It’s cached to
  `mmrag_cache/page_embeddings.npy` afterwards and reload is instant.
- **Pages render but look blurry** — bump `RENDER_DPI` in the notebook
  config cell from 200 → 300.

---

## 12. After your session

`$SCRATCH` is **periodically purged** of files untouched for ~10 days
(check the current policy with `showquota` on the cluster). For
anything you want to keep long-term:

```bash
# back up the artifacts to your group's project storage if you have one
cp -r $SCRATCH/MRAG/mmrag_cache  /scratch/group/<your-project>/MRAG/
# or back to your laptop
rsync -avh <NetID>@grace-dtn1.hprc.tamu.edu:/scratch/user/<NetID>/MRAG/mmrag_cache ./
```

You're done. From now on a typical session is just:
**OnDemand → JupyterLab (Anaconda3 + WebProxy + mrag env + 1 GPU) →
open `MUTCD_MRAG_HPRC.ipynb` → run all → ask questions.**
