# RAGBench manual download (Zscaler / corporate network)

Automated `rag-kag download-data` uses Hugging Face xet CDN. On Zuora laptops that CDN is MITM'd and fails with `403` or `CERTIFICATE_VERIFY_FAILED`.

**Workaround:** download parquet files in a browser, or use **[download_ragbench_colab.ipynb](download_ragbench_colab.ipynb)** on Google Colab (downloads all phase-1.1 subsets + embedder to Drive), then copy to `data/raw/ragbench/`.

## Phase 1.1 representative subsets

| Subset | Domain | HF folder |
|--------|--------|-----------|
| covidqa | biomedical | [covidqa](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/covidqa) |
| pubmedqa | biomedical | [pubmedqa](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/pubmedqa) |
| hotpotqa | general | [hotpotqa](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/hotpotqa) |
| msmarco | general | [msmarco](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/msmarco) |
| cuad | legal | [cuad](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/cuad) |
| emanual | customer_support | [emanual](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/emanual) |
| techqa | customer_support | [techqa](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/techqa) |
| finqa | finance | [finqa](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/finqa) |
| tatqa | finance | [tatqa](https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/tatqa) |

For each subset, download at least `train-00000-of-00001.parquet` (validation/test optional for EDA).

## Target layout

```
data/raw/ragbench/
  covidqa/train-00000-of-00001.parquet
  pubmedqa/train-00000-of-00001.parquet
  ...
```

## Verify

```bash
rag-kag data-status
rag-kag eda --output docs/eda/ragbench_eda.md --json experiments/eda/ragbench_stats.json
```
