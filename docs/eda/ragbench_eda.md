# RAGBench EDA — Phase 1.1

Generated: 2026-06-20 06:28 UTC
Split: `train`
Limit: full split

## Subset coverage

Representative subsets (one or two per RAGBench domain):

- `covidqa` → biomedical
- `pubmedqa` → biomedical
- `hotpotqa` → general
- `msmarco` → general
- `cuad` → legal
- `emanual` → customer_support
- `techqa` → customer_support
- `finqa` → finance
- `tatqa` → finance

## Document length distribution

Word counts use whitespace tokenization (same as the sliding-window chunker).
`pct_gt_256` = % of documents longer than the V1 baseline chunk_size (256 words). When this is 0%, V1 retrieval is degenerate (every doc fits in one chunk).

| subset | domain | n | docs | doc median | doc max | >64% | >128% | >256% | >512% | rec chunk | overlap | local |
|--------|--------|---|------|------------|---------|------|-------|-------|-------|-----------|---------|-------|
| covidqa | biomedical | 1252 | 5008 | 106 | 136 | 78.2 | 0.2 | 0.0 | 0.0 | 72 | 18 | yes |
| pubmedqa | biomedical | 19600 | 98000 | 45 | 375 | 28.7 | 3.4 | 0.1 | 0.0 | 40 | 10 | yes |
| hotpotqa | general | 1883 | 7515 | 74 | 705 | 59.4 | 14.1 | 0.8 | 0.1 | 60 | 15 | yes |
| msmarco | general | 1870 | 15275 | 70 | 188 | 54.9 | 1.3 | 0.0 | 0.0 | 48 | 12 | yes |
| cuad | legal | 1530 | 1530 | 5420 | 47733 | 100.0 | 99.7 | 98.7 | 95.4 | 3014 | 64 | yes |
| emanual | customer_support | 1054 | 3162 | 105 | 1029 | 64.3 | 43.9 | 17.3 | 4.0 | 56 | 14 | yes |
| techqa | customer_support | 1192 | 5960 | 374 | 6564 | 99.1 | 92.4 | 73.1 | 29.9 | 273 | 64 | yes |
| finqa | finance | 12502 | 36228 | 128 | 1489 | 67.8 | 49.9 | 36.0 | 16.1 | 52 | 13 | yes |
| tatqa | finance | 26430 | 126300 | 47 | 636 | 33.6 | 7.3 | 0.6 | 0.0 | 32 | 8 | yes |

## Per-subset detail

### covidqa (biomedical)
- Examples: 1,252; documents: 5,008 (mean 4.0 docs/example)
- Doc words: min=7, mean=90.4, median=106.0, p90=119, p99=125, max=136
- Question words (mean): 9.4
- Sentence keys per example: relevant=5.1, utilized=3.0
- Median sentence length: 18.0 words
- Recommended `chunk_size` / `overlap`: 72 / 18 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.288, utilization=0.169, completeness=0.636, adherence=0.852

### pubmedqa (biomedical)
- Examples: 19,600; documents: 98,000 (mean 5.0 docs/example)
- Doc words: min=1, mean=52.6, median=45.0, p90=98, p99=164, max=375
- Question words (mean): 13.4
- Sentence keys per example: relevant=6.7, utilized=4.3
- Median sentence length: 20.0 words
- Recommended `chunk_size` / `overlap`: 40 / 10 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.565, utilization=0.380, completeness=0.655, adherence=0.755

### hotpotqa (general)
- Examples: 1,883; documents: 7,515 (mean 4.0 docs/example)
- Doc words: min=7, mean=82.9, median=74.0, p90=142, p99=239, max=705
- Question words (mean): 15.9
- Sentence keys per example: relevant=3.1, utilized=2.2
- Median sentence length: 20.0 words
- Recommended `chunk_size` / `overlap`: 60 / 15 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.221, utilization=0.153, completeness=0.741, adherence=0.916

### msmarco (general)
- Examples: 1,870; documents: 15,275 (mean 8.2 docs/example)
- Doc words: min=10, mean=70.5, median=70.0, p90=104, p99=132, max=188
- Question words (mean): 6.0
- Sentence keys per example: relevant=12.0, utilized=6.4
- Median sentence length: 16.0 words
- Recommended `chunk_size` / `overlap`: 48 / 12 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.386, utilization=0.211, completeness=0.556, adherence=0.872

### cuad (legal)
- Examples: 1,530; documents: 1,530 (mean 1.0 docs/example)
- Doc words: min=109, mean=8356.6, median=5420.5, p90=18871, p99=41962, max=47733
- Question words (mean): 21.9
- Sentence keys per example: relevant=9.9, utilized=3.9
- Median sentence length: 22.0 words
- Recommended `chunk_size` / `overlap`: 3014 / 64 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.096, utilization=0.046, completeness=0.770, adherence=0.916

### emanual (customer_support)
- Examples: 1,054; documents: 3,162 (mean 3.0 docs/example)
- Doc words: min=6, mean=156.7, median=105.0, p90=327, p99=915, max=1029
- Question words (mean): 8.5
- Sentence keys per example: relevant=6.4, utilized=5.0
- Median sentence length: 14.0 words
- Recommended `chunk_size` / `overlap`: 56 / 14 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.258, utilization=0.207, completeness=0.809, adherence=0.855

### techqa (customer_support)
- Examples: 1,192; documents: 5,960 (mean 5.0 docs/example)
- Doc words: min=32, mean=458.7, median=374.0, p90=843, p99=1583, max=6564
- Question words (mean): 52.2
- Sentence keys per example: relevant=13.2, utilized=7.8
- Median sentence length: 7.0 words
- Recommended `chunk_size` / `overlap`: 273 / 64 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.067, utilization=0.041, completeness=0.657, adherence=0.626

### finqa (finance)
- Examples: 12,502; documents: 36,228 (mean 2.9 docs/example)
- Doc words: min=1, mean=238.4, median=128.0, p90=602, p99=951, max=1489
- Question words (mean): 16.6
- Sentence keys per example: relevant=1.5, utilized=1.2
- Median sentence length: 26.0 words
- Recommended `chunk_size` / `overlap`: 52 / 13 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.078, utilization=0.067, completeness=0.916, adherence=0.919

### tatqa (finance)
- Examples: 26,430; documents: 126,300 (mean 4.8 docs/example)
- Doc words: min=1, mean=57.1, median=47.0, p90=114, p99=227, max=636
- Question words (mean): 12.5
- Sentence keys per example: relevant=2.0, utilized=1.3
- Median sentence length: 23.0 words
- Recommended `chunk_size` / `overlap`: 32 / 8 (p30 doc length + sentence alignment)
- Dataset reference TRACe (mean): relevance=0.303, utilization=0.210, completeness=0.790, adherence=0.958

## Schema notes

Key RAGBench fields used by the pipeline:

- `question` — user query
- `documents` / `documents_sentences` — retrieved corpus with sentence keys (`0a`, `1b`, …)
- `all_relevant_sentence_keys` / `all_utilized_sentence_keys` — TRACe ground truth
- `relevance_score`, `utilization_score`, `completeness_score`, `adherence_score` — reference scores for evaluator validation

## Chunk size methodology

1. Target ≥50% of documents producing >1 chunk at chosen `chunk_size` (use p30 of doc lengths).
2. Align to sentence boundaries via median sentence length.
3. Keep under embedder context (~512 tokens ≈ ~350 words for bge-small).
