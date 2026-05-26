"""Mixed-effects robustness check for the scaled TCD experiment.

The main experiment reuses names, professions, and templates, so row-level
tests overstate independence. This script fits a hierarchical robustness model
that treats those repeated design factors as random intercepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "results" / "30_scaled" / "raw_rows.json"
OUT = ROOT / "results" / "30_scaled" / "mixed_effects.json"


def load_rows() -> pd.DataFrame:
    rows = json.loads(ROWS.read_text())
    df = pd.DataFrame(rows)
    df["name_pair"] = df["name_a"] + "_" + df["name_b"]
    df["template"] = "T" + df["template_id"].astype(str)
    df["func_minus_cont"] = df["func_tcd"] - df["cont_tcd"]
    return df


def fit_long_model(df: pd.DataFrame) -> dict:
    """Fit TCD ~ is_function with crossed random intercepts.

    Long format retains the direct interpretation of the fixed effect:
    the extra mean absolute TCD assigned to function words after accounting
    for repeated name-pair, profession, and template factors.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    long_df = pd.concat(
        [
            df[["name_pair", "profession", "template"]].assign(
                tcd=df["func_tcd"], is_function=1
            ),
            df[["name_pair", "profession", "template"]].assign(
                tcd=df["cont_tcd"], is_function=0
            ),
        ],
        ignore_index=True,
    )

    model = MixedLM.from_formula(
        "tcd ~ is_function",
        data=long_df,
        groups=long_df["name_pair"],
        re_formula="~1",
        vc_formula={
            "profession": "0 + C(profession)",
            "template": "0 + C(template)",
        },
    )
    result = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
    ci = result.conf_int().loc["is_function"]
    return {
        "model": "long_tcd_random_intercepts",
        "n_observations": int(len(long_df)),
        "n_name_pairs": int(long_df["name_pair"].nunique()),
        "n_professions": int(long_df["profession"].nunique()),
        "n_templates": int(long_df["template"].nunique()),
        "beta_is_function": float(result.params["is_function"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(result.pvalues["is_function"]),
        "converged": bool(result.converged),
        "aic": float(result.aic) if result.aic == result.aic else None,
    }


def fit_difference_model(df: pd.DataFrame) -> dict:
    """Fallback and sensitivity model on paired Func-Cont differences."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    model = MixedLM.from_formula(
        "func_minus_cont ~ 1",
        data=df,
        groups=df["name_pair"],
        re_formula="~1",
        vc_formula={
            "profession": "0 + C(profession)",
            "template": "0 + C(template)",
        },
    )
    result = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
    ci = result.conf_int().loc["Intercept"]
    return {
        "model": "paired_difference_random_intercepts",
        "n_observations": int(len(df)),
        "mean_func_minus_cont": float(df["func_minus_cont"].mean()),
        "intercept": float(result.params["Intercept"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(result.pvalues["Intercept"]),
        "converged": bool(result.converged),
        "aic": float(result.aic) if result.aic == result.aic else None,
    }


def main() -> None:
    df = load_rows()
    output = {
        "input": str(ROWS.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "overall": {
            "mean_func_tcd": float(df["func_tcd"].mean()),
            "mean_cont_tcd": float(df["cont_tcd"].mean()),
            "mean_func_minus_cont": float(df["func_minus_cont"].mean()),
            "ratio": float(df["func_tcd"].mean() / df["cont_tcd"].mean()),
        },
    }

    try:
        output["long_model"] = fit_long_model(df)
    except Exception as exc:  # pragma: no cover - diagnostic fallback path
        output["long_model_error"] = repr(exc)

    try:
        output["difference_model"] = fit_difference_model(df)
    except Exception as exc:  # pragma: no cover - diagnostic fallback path
        output["difference_model_error"] = repr(exc)

    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
