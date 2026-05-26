"""
Local remaining experiments: attention rollout, SPLADE, CrossEncoder.
"""
import json, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import classify_token, compute_tcd, compute_func_cont_tcd
from src.metrics.stats import mann_whitney_one_sided

TEMPLATE = "{NAME} has over ten years of experience in this field."
NAME_PAIRS = [("Emily","Lakisha"),("Greg","Jamal"),("Sarah","Tanisha"),
              ("Todd","Tyrone"),("Brad","Darnell")]

model = ColBERTRetriever(device="mps")
model.load()

# ══════════════════════════════════════════════════════════════════════════
# EXP A: Attention Rollout
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP A: Attention Rollout — name→function vs name→content flow")
print("="*60)

from transformers import AutoModel
attn_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0",
                                        attn_implementation="eager").to("mps")
attn_model.eval()
print("  ✓ Eager-attention model loaded for rollout")

def compute_rollout(text, is_query=False):
    """Attention rollout (Abnar & Zuidema 2020)."""
    import torch
    prefix = "query: " if is_query else "document: "
    inputs = model.tokenizer(prefix + text, return_tensors="pt",
                              truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = attn_model(**inputs, output_attentions=True)
    attentions = torch.stack(outputs.attentions)  # (n_layers, batch, heads, seq, seq)
    att_avg = attentions[:, 0].mean(dim=1).cpu().numpy()  # (n_layers, seq, seq)
    n_layers, seq_len, _ = att_avg.shape
    rollout = np.eye(seq_len)
    for l in range(n_layers):
        a = 0.5 * np.eye(seq_len) + 0.5 * att_avg[l]
        a = a / a.sum(axis=-1, keepdims=True)
        rollout = rollout @ a
    return rollout

rollout_results = []
for na, nb in NAME_PAIRS:
    for name in [na, nb]:
        doc = TEMPLATE.replace("{NAME}", name)
        rollout = compute_rollout(doc)
        tokens = model.get_tokens(doc, is_query=False)

        # Find name token position (skip [CLS], document, :)
        name_idx = None
        for i, tok in enumerate(tokens):
            clean = tok.replace("##", "").lower()
            if clean == name.lower()[:len(clean)] and classify_token(tok) == "content" and tok not in ["document", ":"]:
                name_idx = i
                break
        if name_idx is None:
            # fallback: first content token after prefix
            for i, tok in enumerate(tokens):
                if i > 2 and classify_token(tok) == "content":
                    name_idx = i
                    break
        if name_idx is None:
            continue

        attn_from_name = rollout[:, name_idx]
        func_attn, cont_attn = [], []
        for i, tok in enumerate(tokens):
            if i == name_idx: continue
            cls = classify_token(tok)
            if cls == "function":
                func_attn.append(attn_from_name[i])
            elif cls == "content":
                cont_attn.append(attn_from_name[i])

        if func_attn and cont_attn:
            rollout_results.append({
                "name": name,
                "name_idx": name_idx,
                "name_token": tokens[name_idx],
                "mean_func_attn": float(np.mean(func_attn)),
                "mean_cont_attn": float(np.mean(cont_attn)),
                "ratio": float(np.mean(func_attn) / np.mean(cont_attn)),
                "n_func": len(func_attn),
                "n_cont": len(cont_attn),
            })

if rollout_results:
    ratios = [r["ratio"] for r in rollout_results]
    func_all = [r["mean_func_attn"] for r in rollout_results]
    cont_all = [r["mean_cont_attn"] for r in rollout_results]
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(func_all, cont_all, alternative="greater")
    print(f"  Mean func attention from name: {np.mean(func_all):.4f}")
    print(f"  Mean cont attention from name: {np.mean(cont_all):.4f}")
    print(f"  Ratio: {np.mean(ratios):.2f}×")
    print(f"  Wilcoxon p = {p:.6f}")
    print(f"  {'✓' if p < 0.05 else '✗'} Names route more attention to function words")

with open(RESULTS_DIR / "attention_rollout.json", "w") as f:
    json.dump(rollout_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# EXP B: Identity Probing
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP B: Identity Probing — per-position linear classifier")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

TEMPLATES_PROBE = [
    "{NAME} has over ten years of experience in this field.",
    "{NAME} is a dedicated professional with strong credentials.",
    "{NAME} brings extensive expertise and proven track record.",
    "{NAME} holds advanced certifications in their area.",
    "{NAME} has worked with leading organisations across the industry.",
]

embeddings_by_pos = {}
tokens_at_pos = {}

for tmpl in TEMPLATES_PROBE:
    for label, names in enumerate([["Emily","Greg","Sarah","Todd","Brad"],
                                    ["Lakisha","Jamal","Tanisha","Tyrone","Darnell"]]):
        for name in names:
            doc = tmpl.replace("{NAME}", name)
            emb = model.encode(doc, is_query=False).cpu().numpy()
            tokens = model.get_tokens(doc, is_query=False)
            for pos in range(min(len(tokens), emb.shape[0])):
                if pos not in embeddings_by_pos:
                    embeddings_by_pos[pos] = []
                    tokens_at_pos[pos] = tokens[pos]
                embeddings_by_pos[pos].append((emb[pos], label))

probe_results = []
for pos in sorted(embeddings_by_pos.keys()):
    data = embeddings_by_pos[pos]
    if len(data) < 10: continue
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    if len(set(y)) < 2: continue
    clf = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, y, cv=min(5, len(data)//2), scoring="accuracy")
    tok = tokens_at_pos.get(pos, f"[{pos}]")
    tok_class = classify_token(tok)
    probe_results.append({
        "pos": pos, "token": tok, "class": tok_class,
        "accuracy": float(np.mean(scores)), "std": float(np.std(scores)),
        "n": len(data),
    })

func_probes = [r["accuracy"] for r in probe_results if r["class"] == "function"]
cont_probes = [r["accuracy"] for r in probe_results if r["class"] == "content"]

if func_probes and cont_probes:
    mw_probe = mann_whitney_one_sided(np.array(func_probes), np.array(cont_probes), "greater")
    print(f"  Function-word probe accuracy: {np.mean(func_probes):.3f} ± {np.std(func_probes):.3f}")
    print(f"  Content-word probe accuracy:  {np.mean(cont_probes):.3f} ± {np.std(cont_probes):.3f}")
    print(f"  MW p = {mw_probe['p_value']:.4f}")
    print(f"  {'✓' if mw_probe['p_value'] < 0.05 else '✗'} Function words encode more identity")
    print("\n  Per-position breakdown:")
    for r in probe_results:
        marker = "★" if r["accuracy"] > 0.7 else " "
        print(f"    {marker} pos={r['pos']:2d} {r['token']:12s} [{r['class']:8s}] acc={r['accuracy']:.3f}")

with open(RESULTS_DIR / "identity_probing.json", "w") as f:
    json.dump(probe_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# EXP C: SPLADE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP C: SPLADE comparison")
print("="*60)

QUERIES = ["Who is a qualified doctor?", "Who is a skilled engineer?",
           "Who is a competent lawyer?"]

try:
    from src.models.splade import SPLADERetriever
    splade = SPLADERetriever(device="mps")
    splade.load()
    splade_ss = []
    for q in QUERIES:
        for na, nb in NAME_PAIRS[:3]:
            da = TEMPLATE.replace("{NAME}", na)
            db = TEMPLATE.replace("{NAME}", nb)
            cf = splade.counterfactual_score(q, da, db)
            splade_ss.append(cf.score_sensitivity)
    print(f"  SPLADE mean SS: {np.mean(splade_ss):.4f} (n={len(splade_ss)})")
    with open(RESULTS_DIR / "splade_results.json", "w") as f:
        json.dump({"mean_ss": float(np.mean(splade_ss)), "n": len(splade_ss),
                   "values": [float(v) for v in splade_ss]}, f, indent=2)
except Exception as e:
    print(f"  SPLADE failed: {e}")
    with open(RESULTS_DIR / "splade_results.json", "w") as f:
        json.dump({"error": str(e)}, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# EXP D: Cross-Encoder
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP D: Cross-Encoder comparison")
print("="*60)

try:
    from src.models.cross_encoder import CrossEncoderRetriever
    ce = CrossEncoderRetriever(device="mps")
    ce.load()
    ce_ss = []
    for q in QUERIES:
        for na, nb in NAME_PAIRS[:3]:
            da = TEMPLATE.replace("{NAME}", na)
            db = TEMPLATE.replace("{NAME}", nb)
            cf = ce.counterfactual_score(q, da, db)
            ce_ss.append(cf.score_sensitivity)
    print(f"  CrossEncoder mean SS: {np.mean(ce_ss):.4f} (n={len(ce_ss)})")
    with open(RESULTS_DIR / "cross_encoder_results.json", "w") as f:
        json.dump({"mean_ss": float(np.mean(ce_ss)), "n": len(ce_ss),
                   "values": [float(v) for v in ce_ss]}, f, indent=2)
except Exception as e:
    print(f"  CrossEncoder failed: {e}")
    with open(RESULTS_DIR / "cross_encoder_results.json", "w") as f:
        json.dump({"error": str(e)}, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL REMAINING EXPERIMENTS COMPLETE")
print("="*60)
