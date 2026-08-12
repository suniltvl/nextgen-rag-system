# TRACe validator — covidqa (train)

Examples: 1252

| metric | matches | avg ours | avg reference | avg delta |
|--------|---------|----------|---------------|-----------|
| context_relevance | 1249/1252 | 0.2869 | 0.2877 | -0.0008 |
| context_utilization | 1247/1252 | 0.1690 | 0.1695 | -0.0004 |
| completeness | 1252/1252 | 0.6362 | 0.6362 | +0.0000 |
| adherence | 1252/1252 | 0.8522 | 0.8522 | +0.0000 |

## Sample mismatches (first 20)

- `143` context_relevance: ours=0.0000 ref=0.4091 (Δ=-0.4091)
- `143` context_utilization: ours=0.0000 ref=0.1818 (Δ=-0.1818)
- `1633` context_utilization: ours=0.0000 ref=0.0625 (Δ=-0.0625)
- `1761` context_relevance: ours=0.0000 ref=0.3636 (Δ=-0.3636)
- `1761` context_utilization: ours=0.0000 ref=0.0909 (Δ=-0.0909)
- `1246` context_relevance: ours=0.0000 ref=0.2500 (Δ=-0.2500)
- `1246` context_utilization: ours=0.0000 ref=0.0833 (Δ=-0.0833)
- `1101` context_utilization: ours=0.0000 ref=0.1250 (Δ=-0.1250)
