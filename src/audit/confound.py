"""
Confound decomposition via matched-pair regression.

Implements the OLS-based decomposition used in P2/P2b:
  SS ~ β_race * cross_race + β_gender * cross_gender
       + β_token * token_gap + β_freq * freq_gap + β_rarity * both_rare
       + profession_FE + template_FE

with clustered standard errors by profession.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_design_matrix(
    pair_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build binary features for confound decomposition regression.

    Expected columns in pair_data:
      - name_a, name_b: name strings
      - race_a, race_b, gender_a, gender_b: categorical
      - bpe_a, bpe_b: int BPE token counts
      - freq_a, freq_b: float within-race frequency
      - profession: string
      - template: string
      - ss: float Score Sensitivity
      - func_tcd: float Func-TCD

    Returns DataFrame with added binary feature columns.
    """
    df = pair_data.copy()

    # Binary features
    df["cross_race"] = (df["race_a"] != df["race_b"]).astype(int)
    df["cross_gender"] = (df["gender_a"] != df["gender_b"]).astype(int)
    df["token_gap"] = abs(df["bpe_a"] - df["bpe_b"])
    df["freq_gap"] = abs(df["freq_a"] - df["freq_b"])
    df["both_rare"] = (
        (df["bpe_a"] > 2) & (df["bpe_b"] > 2)
    ).astype(int)

    return df


def run_nested_models(
    df: pd.DataFrame,
    outcome: str = "func_tcd",
    cluster_col: str = "profession",
) -> List[Dict]:
    """Run nested regression models for R² progression analysis.

    Models (cumulative):
      M0: outcome ~ cross_race
      M1: outcome ~ cross_race + cross_gender
      M2: outcome ~ cross_race + cross_gender + token_gap
      M3: outcome ~ cross_race + cross_gender + token_gap + freq_gap
      M4: outcome ~ cross_race + cross_gender + token_gap + freq_gap + both_rare

    Returns list of dicts with model specs and R² values.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels required for regression. "
            "Install with: pip install statsmodels"
        )

    features_sequence = [
        ["cross_race"],
        ["cross_race", "cross_gender"],
        ["cross_race", "cross_gender", "token_gap"],
        ["cross_race", "cross_gender", "token_gap", "freq_gap"],
        ["cross_race", "cross_gender", "token_gap", "freq_gap", "both_rare"],
    ]

    results = []
    for i, features in enumerate(features_sequence):
        X = sm.add_constant(df[features].astype(float))
        y = df[outcome].astype(float)

        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cluster_col]},
        )

        coefs = {}
        for feat in features:
            coefs[feat] = {
                "coefficient": float(model.params[feat]),
                "std_error": float(model.bse[feat]),
                "p_value": float(model.pvalues[feat]),
                "ci_lower": float(model.conf_int().loc[feat, 0]),
                "ci_upper": float(model.conf_int().loc[feat, 1]),
            }

        results.append({
            "model_id": f"M{i}",
            "features": features,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "n_obs": int(model.nobs),
            "coefficients": coefs,
        })

    return results


def compute_r2_progression(
    nested_results: List[Dict],
) -> List[Dict[str, float]]:
    """Compute incremental R² gains from nested model sequence.

    Returns list of {feature_added, r2, delta_r2} dicts.
    """
    progression = []
    for i, res in enumerate(nested_results):
        delta = res["r_squared"] - (
            nested_results[i - 1]["r_squared"] if i > 0 else 0.0
        )
        new_feat = (
            res["features"][-1]
            if i > 0
            else res["features"][0]
        )
        progression.append({
            "model": res["model_id"],
            "feature_added": new_feat,
            "r_squared": res["r_squared"],
            "delta_r_squared": delta,
        })
    return progression
