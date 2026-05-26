"""
Cluster bootstrap robustness checks for the 55,440-row scaled experiment.

The main experiment reuses name pairs, professions, and templates, so row-level
tests overstate independence. This script resamples whole clusters and reports
confidence intervals for the Func/Cont TCD ratio.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_ROWS = ROOT / "results" / "30_scaled" / "raw_rows.json"
OUT = ROOT / "results" / "30_scaled" / "cluster_bootstrap.json"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def ratio(rows):
    return mean([r["func_tcd"] for r in rows]) / mean([r["cont_tcd"] for r in rows])


def cluster_key(row, cluster):
    if cluster == "name_pair":
        return tuple(sorted((row["name_a"], row["name_b"])))
    if cluster == "profession":
        return row["profession"]
    if cluster == "template":
        return row["template_id"]
    if cluster == "pair_profession":
        return (tuple(sorted((row["name_a"], row["name_b"]))), row["profession"])
    raise ValueError(f"Unknown cluster: {cluster}")


def bootstrap(rows, cluster, n_boot=2000, seed=13):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[cluster_key(row, cluster)].append(row)

    keys = list(groups)
    samples = []
    for _ in range(n_boot):
        sampled_rows = []
        for _ in keys:
            sampled_rows.extend(groups[rng.choice(keys)])
        samples.append(ratio(sampled_rows))

    cluster_ratios = [ratio(groups[k]) for k in keys]
    return {
        "n_clusters": len(keys),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "median": quantile(samples, 0.5),
        "pct_ratio_gt_1_by_cluster": sum(x > 1.0 for x in cluster_ratios) / len(cluster_ratios),
    }


def main():
    with RAW_ROWS.open() as f:
        rows = json.load(f)

    result = {
        "n_rows": len(rows),
        "overall_ratio": ratio(rows),
        "bootstraps": {
            cluster: bootstrap(rows, cluster)
            for cluster in ["name_pair", "profession", "template", "pair_profession"]
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
