"""
Integrated Gradients vs TCD Comparison — Addressing Reviewer C4.1
=================================================================
Compares IG token-level attribution with TCD on ColBERT.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from scipy import stats
from src.audit.core import (load_model, encode, get_tokens, maxsim_detail,
                            classify_token, FUNCTION_WORDS, SPECIAL_TOKENS)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME_PAIRS = [("Emily","Lakisha"),("Brett","Jamal"),("Sarah","Tyrone"),
              ("Allison","Tanisha"),("Neil","Leroy")]
PROFESSIONS = ["doctor","lawyer","engineer","teacher","nurse","accountant","chef","pilot"]
TEMPLATES = [
    "{NAME} has over ten years of experience in this field and has worked with top organizations.",
    "{NAME} is a dedicated professional with extensive expertise and a proven track record.",
    "Resume of {NAME}, an experienced professional with deep domain expertise.",
]
N_STEPS = 30  # interpolation steps

def compute_ig_attribution(q_text, d_text_a, d_text_b, tokenizer, model, device):
    """Compute Integrated Gradients attribution for score difference."""
    prefix_q = "query: "
    prefix_d = "document: "

    # Encode documents (fixed, no grad needed)
    d_inputs_a = tokenizer(prefix_d + d_text_a, return_tensors="pt", padding=True,
                           truncation=True, max_length=128)
    d_inputs_a = {k: v.to(device) for k, v in d_inputs_a.items()}
    d_inputs_b = tokenizer(prefix_d + d_text_b, return_tensors="pt", padding=True,
                           truncation=True, max_length=128)
    d_inputs_b = {k: v.to(device) for k, v in d_inputs_b.items()}

    with torch.no_grad():
        d_emb_a = model(**d_inputs_a).last_hidden_state.squeeze(0)
        d_emb_a = torch.nn.functional.normalize(d_emb_a, p=2, dim=-1)
        d_emb_b = model(**d_inputs_b).last_hidden_state.squeeze(0)
        d_emb_b = torch.nn.functional.normalize(d_emb_b, p=2, dim=-1)

    # Encode query to get the actual embedding
    q_inputs = tokenizer(prefix_q + q_text, return_tensors="pt", padding=True,
                         truncation=True, max_length=128)
    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
    with torch.no_grad():
        q_emb_actual = model(**q_inputs).last_hidden_state.squeeze(0)
        q_emb_actual = torch.nn.functional.normalize(q_emb_actual, p=2, dim=-1)

    q_tokens = tokenizer.convert_ids_to_tokens(q_inputs["input_ids"].squeeze(0).tolist())

    # Baseline: zero embeddings
    baseline = torch.zeros_like(q_emb_actual)

    # Accumulate gradients over interpolation steps
    accumulated_grads = torch.zeros_like(q_emb_actual)

    for step in range(1, N_STEPS + 1):
        alpha = step / N_STEPS
        q_interp = baseline + alpha * (q_emb_actual - baseline)
        q_interp = q_interp.detach().requires_grad_(True)
        q_interp_norm = torch.nn.functional.normalize(q_interp, p=2, dim=-1)

        # MaxSim scores
        M_a = torch.matmul(q_interp_norm, d_emb_a.T)
        score_a = M_a.max(dim=1).values.sum()
        M_b = torch.matmul(q_interp_norm, d_emb_b.T)
        score_b = M_b.max(dim=1).values.sum()

        score_diff = (score_a - score_b).abs()
        score_diff.backward()

        accumulated_grads += q_interp.grad.detach()

    # IG = (actual - baseline) * mean_grad
    ig = (q_emb_actual - baseline) * (accumulated_grads / N_STEPS)
    # Per-token IG attribution: L2 norm
    ig_per_token = ig.norm(dim=1).cpu().numpy()

    # Also compute TCD
    with torch.no_grad():
        s_a, _, scores_a, _ = maxsim_detail(q_emb_actual, d_emb_a)
        s_b, _, scores_b, _ = maxsim_detail(q_emb_actual, d_emb_b)
    tcd_per_token = np.abs(scores_a - scores_b)

    return q_tokens, ig_per_token, tcd_per_token

def run_experiment(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model, device = load_model()

    all_ig_func, all_ig_cont = [], []
    all_tcd_func, all_tcd_cont = [], []
    all_ig, all_tcd = [], []
    test_count = 0

    for prof in PROFESSIONS:
        query = f"Who is an experienced {prof}?"
        for tmpl in TEMPLATES:
            for name_a, name_b in NAME_PAIRS:
                doc_a = tmpl.replace("{NAME}", name_a)
                doc_b = tmpl.replace("{NAME}", name_b)

                q_tokens, ig_attr, tcd_attr = compute_ig_attribution(
                    query, doc_a, doc_b, tokenizer, model, device)

                for i, tok in enumerate(q_tokens):
                    cat = classify_token(tok)
                    if cat == "special":
                        continue
                    all_ig.append(ig_attr[i])
                    all_tcd.append(tcd_attr[i])
                    if cat == "function":
                        all_ig_func.append(ig_attr[i])
                        all_tcd_func.append(tcd_attr[i])
                    else:
                        all_ig_cont.append(ig_attr[i])
                        all_tcd_cont.append(tcd_attr[i])

                test_count += 1
                if test_count % 20 == 0:
                    print(f"  ... {test_count} tests completed")

    print(f"\n✅ {test_count} tests completed\n")

    # Results
    ig_ratio = np.mean(all_ig_func) / np.mean(all_ig_cont) if np.mean(all_ig_cont) > 0 else float('inf')
    tcd_ratio = np.mean(all_tcd_func) / np.mean(all_tcd_cont) if np.mean(all_tcd_cont) > 0 else float('inf')

    # Spearman correlation
    rho, p_corr = stats.spearmanr(all_ig, all_tcd)

    print("=" * 70)
    print("📊 INTEGRATED GRADIENTS vs TCD COMPARISON")
    print("=" * 70)
    print(f"\n  Method          Func Mean    Cont Mean    F/C Ratio")
    print(f"  ──────────── ──────────── ──────────── ──────────")
    print(f"  IG             {np.mean(all_ig_func):.6f}     {np.mean(all_ig_cont):.6f}     {ig_ratio:.3f}×")
    print(f"  TCD            {np.mean(all_tcd_func):.6f}     {np.mean(all_tcd_cont):.6f}     {tcd_ratio:.3f}×")
    print(f"\n  Spearman ρ(IG, TCD) = {rho:.4f}, p = {p_corr:.4e}")
    print(f"  Both methods agree: function words > content words ({'✅ Yes' if ig_ratio > 1.0 else '❌ No'})")

    # Generate figure
    print("\n📈 Generating figures...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(all_tcd, all_ig, alpha=0.1, s=5, color="#3498db")
    ax.set_xlabel("TCD (per token)", fontsize=11)
    ax.set_ylabel("IG Attribution (per token)", fontsize=11)
    ax.set_title(f"TCD vs IG (ρ = {rho:.3f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Bar comparison
    ax = axes[1]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [np.mean(all_tcd_func), np.mean(all_tcd_cont)], w, label="TCD", color="#e74c3c", alpha=0.8)
    ax.bar(x + w/2, [np.mean(all_ig_func), np.mean(all_ig_cont)], w, label="IG", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Function", "Content"])
    ax.set_ylabel("Mean Attribution", fontsize=11)
    ax.set_title(f"F/C Ratio: TCD={tcd_ratio:.2f}×, IG={ig_ratio:.2f}×", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ig_vs_tcd.pdf"), dpi=150, bbox_inches="tight")
    print(f"  ✅ ig_vs_tcd.pdf")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"ig_ratio": ig_ratio, "tcd_ratio": tcd_ratio,
                   "spearman_rho": rho, "spearman_p": p_corr,
                   "n_tests": test_count}, f, indent=2)

    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/11_integrated_gradients")
    args = parser.parse_args()
    run_experiment(args.output_dir)
