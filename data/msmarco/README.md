MS MARCO passage collection
===========================

The benchmark-scale real-corpus audit expects the MS MARCO passage ranking
collection at:

```text
data/msmarco/collection.tsv
```

This file is intentionally not committed because the full collection is large.
After placing the TSV here, run:

```bash
.venv/bin/python experiments/08_msmarco_counterfactual/msmarco_tcd_audit.py \
  --data-dir data/msmarco \
  --output-dir results/08_msmarco
```

The script first scans for passages containing PERSON entities, then applies
controlled name swaps and reports aggregate TCD/score-sensitivity summaries.
