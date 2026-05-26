"""
Layer-by-Layer Embedding Probing — Addressing Reviewer D4
==========================================================
Validates the 'representational shift' hypothesis by measuring
how document token embeddings change at each BERT layer when
the identity name is swapped.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from scipy import stats
from transformers import AutoTokenizer, AutoModel
from src.audit.core import MODEL_NAME, FUNCTION_WORDS, SPECIAL_TOKENS, get_device
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

def classify_doc_token(tok):
    clean = tok.replace("##","").lower()
    if tok in SPECIAL_TOKENS or tok in ("[CLS]","[SEP]","document",":"):
        return "special"
    if clean in FUNCTION_WORDS:
        return "function"
    return "content"

def encode_with_layers(text, tokenizer, model, device, is_query=False):
    prefix = "query: " if is_query else "document: "
    inputs = tokenizer(prefix + text, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of 13 tensors (embedding + 12 layers), each [1, seq, dim]
    hidden_states = [h.squeeze(0).cpu() for h in outputs.hidden_states]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).tolist())
    return hidden_states, tokens

def find_name_positions(tokens, name, tokenizer):
    name_tokens = tokenizer.tokenize(name.lower())
    # Find where name tokens appear in the token list
    positions = []
    for i in range(len(tokens)):
        match = True
        for j, nt in enumerate(name_tokens):
            if i+j >= len(tokens) or tokens[i+j].lower().replace("##","") != nt.lower().replace("##",""):
                match = False
                break
        if match:
            positions = list(range(i, i+len(name_tokens)))
            break
    return positions

def run_experiment(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"🔧 Device: {device}")
    print("📦 Loading ColBERTv2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("✅ Model loaded!\n")

    # Results: per layer, per token type
    num_layers = 13  # embedding + 12 transformer layers
    layer_shifts = {cat: [[] for _ in range(num_layers)] for cat in ["name","adjacent_func","adjacent_cont","distant_func","distant_cont"]}

    test_count = 0
    for prof in PROFESSIONS:
        for tmpl in TEMPLATES:
            for name_a, name_b in NAME_PAIRS:
                doc_a = tmpl.replace("{NAME}", name_a)
                doc_b = tmpl.replace("{NAME}", name_b)

                hs_a, tok_a = encode_with_layers(doc_a, tokenizer, model, device, is_query=False)
                hs_b, tok_b = encode_with_layers(doc_b, tokenizer, model, device, is_query=False)

                # Find name positions in each version
                name_pos_a = find_name_positions(tok_a, name_a, tokenizer)
                name_pos_b = find_name_positions(tok_b, name_b, tokenizer)

                # Use the shorter token list length
                min_len = min(len(tok_a), len(tok_b))

                # For tokens that are the same in both versions (non-name tokens),
                # compute cosine distance at each layer
                for layer_idx in range(num_layers):
                    ha = hs_a[layer_idx]  # [seq_a, dim]
                    hb = hs_b[layer_idx]  # [seq_b, dim]

                    for pos in range(min_len):
                        tok = tok_a[pos]
                        tok_type = classify_doc_token(tok)
                        if tok_type == "special":
                            continue

                        # Cosine distance between same-position embeddings
                        va = ha[pos]
                        vb = hb[pos]
                        cos_sim = torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
                        cos_dist = 1.0 - cos_sim

                        # Classify by distance from name
                        is_name = pos in name_pos_a or pos in name_pos_b
                        min_dist_to_name = min(
                            [abs(pos - np) for np in name_pos_a] + [abs(pos - np) for np in name_pos_b] + [999]
                        )

                        if is_name:
                            layer_shifts["name"][layer_idx].append(cos_dist)
                        elif min_dist_to_name <= 3:
                            if tok_type == "function":
                                layer_shifts["adjacent_func"][layer_idx].append(cos_dist)
                            else:
                                layer_shifts["adjacent_cont"][layer_idx].append(cos_dist)
                        else:
                            if tok_type == "function":
                                layer_shifts["distant_func"][layer_idx].append(cos_dist)
                            else:
                                layer_shifts["distant_cont"][layer_idx].append(cos_dist)

                test_count += 1

    print(f"✅ {test_count} tests completed\n")

    # Summary
    print("=" * 70)
    print("📊 LAYER-BY-LAYER REPRESENTATIONAL SHIFT")
    print("=" * 70)

    categories = ["name","adjacent_func","adjacent_cont","distant_func","distant_cont"]
    cat_labels = ["Name token","Adjacent func","Adjacent content","Distant func","Distant content"]
    means = {cat: [] for cat in categories}

    print(f"\n  {'Layer':<8}", end="")
    for label in cat_labels:
        print(f"{label:>16}", end="")
    print()
    print("  " + "─" * 88)

    for layer_idx in range(num_layers):
        layer_name = f"Emb" if layer_idx == 0 else f"L{layer_idx}"
        print(f"  {layer_name:<8}", end="")
        for cat in categories:
            vals = layer_shifts[cat][layer_idx]
            m = np.mean(vals) if vals else 0
            means[cat].append(m)
            print(f"{m:>16.6f}", end="")
        print()

    # Key test: at layers 6-12, is adjacent_func > adjacent_cont?
    print(f"\n📊 KEY TEST: Adjacent function vs content word shift (layers 6-12)")
    for layer_idx in range(6, 13):
        func_vals = layer_shifts["adjacent_func"][layer_idx]
        cont_vals = layer_shifts["adjacent_cont"][layer_idx]
        if func_vals and cont_vals:
            u_stat, p_val = stats.mannwhitneyu(func_vals, cont_vals, alternative="greater")
            d = (np.mean(func_vals) - np.mean(cont_vals)) / np.sqrt(
                (np.var(func_vals) + np.var(cont_vals)) / 2) if np.var(func_vals) + np.var(cont_vals) > 0 else 0
            layer_name = f"L{layer_idx}"
            print(f"  {layer_name}: Func={np.mean(func_vals):.6f}, Cont={np.mean(cont_vals):.6f}, "
                  f"ratio={np.mean(func_vals)/np.mean(cont_vals):.3f}×, d={d:.3f}, p={p_val:.4e}")

    # Generate figure
    print("\n📈 Generating figures...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(num_layers))
    xlabels = ["Emb"] + [f"L{i}" for i in range(1, 13)]
    colors = {"name": "#e74c3c", "adjacent_func": "#e67e22", "adjacent_cont": "#3498db",
              "distant_func": "#f39c12", "distant_cont": "#2ecc71"}
    for cat, label in zip(categories, cat_labels):
        ax.plot(x, means[cat], marker="o", label=label, color=colors[cat], linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Distance (identity swap)", fontsize=12)
    ax.set_title("Representational Shift by Layer and Token Type", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "layer_probing.pdf"), dpi=150, bbox_inches="tight")
    print(f"  ✅ layer_probing.pdf")

    # Save data
    results = {cat: [float(np.mean(layer_shifts[cat][l])) if layer_shifts[cat][l] else 0
                     for l in range(num_layers)] for cat in categories}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/10_embedding_probing")
    args = parser.parse_args()
    run_experiment(args.output_dir)
