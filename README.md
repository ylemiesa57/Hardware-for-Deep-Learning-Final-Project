# Final Project — RAG_LLM

Running a RAG workload on a simple FP16 accelerator model.

## What's in here

- `workspace/arch/basic8.yaml` — the architecture (disk, DRAM, SRAM, MAC)
- `workspace/workload/RAG_LLM.yaml` — the RAG einsums
- `workspace/RAG_LLM_analysis.ipynb` — the main notebook

## How to run

```powershell
cd final-project-16
$env:DOCKER_ARCH = "amd64"
docker compose up
```

Open `localhost:8888` and run `RAG_LLM_analysis.ipynb`.

## RAG workload summary

Documents are embedded offline into `Z_D[n,d]` and stored on disk.
At inference, `Z_D` is streamed into DRAM and the query goes through
embed → QKV → attention → similarity search → top-k retrieval.

Default parameters: N=21M docs, T=100 tokens, D=768, K=10.

## Main experiment

`Z_D` at 21M docs is ~32 GB, but DRAM defaults to 8 GB.
We sweep `DRAM_SIZE_GB` to study the cost of caching embeddings in DRAM vs streaming from disk.
