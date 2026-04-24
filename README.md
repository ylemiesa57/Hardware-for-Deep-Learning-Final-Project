# Final Project — Retrieval-Augmented Generation

Modeling a RAG + LLM inference pipeline in **AccelForge** to study where energy and
latency go across the memory hierarchy, and when/how caching retrieved documents
pays off.

MIT 6.5931 — Yaphet Lemiesa (`yaphkl75@mit.edu`), Jocelyn Zhao (`jjz300@mit.edu`).

## What's in here

### Architectures — `workspace/arch/`

| File | Description |
|---|---|
| `basic8.yaml` | Baseline FP16 accelerator: Disk → DRAM → SRAM → MAC. Jinja-parameterized (`DRAM_SIZE_GB`, `MAC_ENERGY_pJ`, clocks, etc.). Default target for the RAG experiments. |
| `EdgeRAG_arch.yaml` | Edge-device variant modeled on Jetson Orin Nano: SD-card tier, LPDDR5 DRAM (34 GB/s), 2 MB SRAM, INT8 MAC @ 0.625 GHz. |
| `basic_analog.yaml` + `_include.yaml` + `_include_functions.py` | Analog compute-in-memory path (RRAM-style) with NeuroSim-style attributes. |
| `memory_cells/*.yaml` | Per-technology parameters (RRAM / SRAM / ECRAM). |

### Workloads — `workspace/workload/`

| File | Description |
|---|---|
| `full_6.7B.yaml` | **Main workload for `M2-Analysis.ipynb`.** Combined encoder + RAG + decoder einsums at 6.7B-decoder scale. |
| `full_EdgeRAG.yaml` | Explicit gte-base encoder + FAISS-style retrieval + Sheared-LLaMA-2.7B decoder, INT8. |
| `RAG.yaml` | Smaller/abstract RAG einsum chain (embedding → QKV → self-attn → `S_score` similarity → top-k). |
| `gpt_175B.yaml` | GPT-scale workload for large-scale sweeps. |
| `examples/*.yaml` | ResNet18 / MobileNet-v3 / ViT for DNN layer sweeps (used by `scripts/utils.run_dnn_layers`). |

### Notebooks — `workspace/`

- **`M2-Analysis.ipynb`** — Milestone 2 analysis: loads `full_6.7B.yaml` onto `basic8`,
  runs each einsum, reports per-component energy/latency and breakdown plots.
- **`M3-Baseline.ipynb`** — Milestone 3 baseline / WIP runs.
- **`visual.ipynb`** — Exploratory AccelForge + graphviz / matplotlib visualizations.

### Supporting Python — `workspace/`

- `_load_spec.py` — `get_spec(arch_name, ...)` loads an arch YAML into an AccelForge
  `Spec` (wires in `arch/_include_functions.py`, optionally inserts a dummy infinite
  main memory for DNN workloads, sets mapper options).
- `scripts/utils.py` — `Result` wrapper (energy, per-component energy, per-compute
  metrics), `quick_run`, `run_dnn_layers`, Jinja/variable bridging for analog CiM
  sweeps.
- `scripts/plots.py` — Matplotlib bar / grouped-bar / stacked-bar helpers (Agg backend).

## How to run

From the repo root:

```powershell
# Windows (amd64)
$env:DOCKER_ARCH = "amd64"
docker compose up
```

```bash
# Linux / Apple Silicon
export DOCKER_ARCH=amd64     # or arm64 on M1/M2
docker compose up
```

This launches the `timeloopaccelergy/accelforge:latest-${DOCKER_ARCH}` image with
the repo mounted at `/home/workspace` and Jupyter on port **8888** (no token).
Open <http://localhost:8888> and run `workspace/M2-Analysis.ipynb`.

## Project overview

RAG (retrieval-augmented generation) augments an LLM by retrieving relevant
documents from an external store and concatenating them into the query. Document
embeddings `Z_D[n,d]` are computed **offline** and stored on disk; at inference:

1. **Encode query** — embed → QKV → self-attention (BERT-base-scale)
2. **Retrieve** — similarity (`SIM`) against `Z_D` → top-k (`TK`) → gather (`II_in`)
3. **Decode** — LLaMA-scale decoder consumes query + retrieved passages

The RAG step adds substantial memory traffic (the corpus is large and far from
compute), which often dominates latency and energy in real deployments.

### Key questions

- Which components (memory vs. compute) dominate energy and latency as retrieval
  workload scales?
- How do document count `N` and document length `T` change the bottleneck?
- Where does caching help, and when does caching cost more energy than it saves?

### Default parameters

`N = 10^5` docs, `T = 512`, `K = 20` retrieved per query, 8 MB SRAM, 8 GB LPDDR5
DRAM, SD-card-backed disk corpus (mirrors EdgeRAG's evaluation platform).

## Experiments

1. **Attention softmax fusion** *(Jocelyn Zhao)* — Compare unfused baseline vs. a
   fused `QK → softmax → AV` mapping that keeps attention scores in the register
   file. Sweep decoder document length `T ∈ {128, 512, 2048, 8192}` at `N = 10^5`.
2. **Relevance-aware DRAM caching** *(Yaphet Lemiesa)* — Replicate EdgeRAG's
   cost-aware LFU policy (evict min of `generation_latency × access_count`). Sweep
   DRAM capacity `∈ {1, 2, 4, 8, 16} GB` over BEIR reuse ratios `1.25 – 4.47`.
3. **Two-tier SRAM + DRAM cache** *(Yaphet Lemiesa)* — Extend experiment 2 to a
   hierarchical cache; study when on-chip SRAM caching pays off for very large
   corpora. Focus on characterizing the regime where SRAM capacity is actually
   large enough to capture the relevant working set (i.e. the hot subset of
   embeddings that get reused across queries), not just when the corpus grows.

All experiments hold PE array dimensions, SRAM capacity, and DRAM capacity
constant unless that dimension is the swept variable. We report energy per query,
latency per query, and energy-delay product.
