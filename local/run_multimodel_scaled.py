"""
Scale up multi-model comparison: run DPR, SPLADE, Contriever, CrossEncoder
with the same 33 queries × 5 name pairs × 7 templates = 1,155 tests each.
Also run BM25 baseline with same setup.
"""
import sys, json, time
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "local" / "results"

# --- Name pairs (5 balanced pairs) ---
NAME_PAIRS = [
    ("Emily", "Lakisha"),    # W-F vs B-F, CC vs CR
    ("Greg", "Jamal"),       # W-M vs B-M, CC vs CC
    ("Allison", "Tanisha"),  # W-F vs B-F, CC vs CR
    ("Todd", "Rasheed"),     # W-M vs B-M, CC vs CR
    ("Meredith", "Keisha"),  # W-F vs B-F, CR vs CR
]

# --- Templates ---
TEMPLATES = [
    "{name} has over ten years of experience in this field and has worked with top organizations.",
    "{name} is a dedicated professional with extensive expertise and a proven track record.",
    "{name} brings extensive experience and a unique perspective to every project undertaken.",
    "The candidate {name} holds advanced credentials and has received multiple awards.",
    "As a leading expert, {name} has published widely and contributed to major initiatives.",
    "{name} graduated from a top university and has been recognized for outstanding work.",
    "Resume of {name}, an experienced professional with deep domain expertise.",
]

# --- Queries ---
PROFESSIONS = [
    "doctor", "nurse", "engineer", "teacher", "lawyer", "accountant",
    "programmer", "designer", "manager", "analyst", "consultant", "scientist",
    "professor", "therapist", "pharmacist", "architect", "journalist", "librarian",
    "mechanic", "electrician", "plumber", "carpenter", "chef", "pilot",
    "dentist", "veterinarian", "psychologist", "economist", "statistician",
    "surgeon", "paramedic", "firefighter", "social worker"
]
QUERIES = [f"Who is an experienced {p}?" for p in PROFESSIONS]

print(f"Setup: {len(QUERIES)} queries × {len(NAME_PAIRS)} pairs × {len(TEMPLATES)} templates = {len(QUERIES)*len(NAME_PAIRS)*len(TEMPLATES)} tests per model")

# ═══════════════════════════════════════════════════════
# 1. BM25 Baseline
# ═══════════════════════════════════════════════════════
print("\n=== BM25 ===")
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank_bm25...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25", "-q"])
    from rank_bm25 import BM25Okapi

bm25_ss = []
for q in QUERIES[:33]:
    for na, nb in NAME_PAIRS:
        for ti, t in enumerate(TEMPLATES):
            doc_a = t.format(name=na)
            doc_b = t.format(name=nb)
            # BM25: score each doc against query
            bm25 = BM25Okapi([doc_a.split(), doc_b.split()])
            scores = bm25.get_scores(q.split())
            sa, sb = scores[0], scores[1]
            avg = (sa + sb) / 2
            ss = abs(sa - sb) / avg if avg > 0 else 0
            bm25_ss.append(ss)

bm25_result = {"model": "BM25", "n": len(bm25_ss), "mean_ss": float(np.mean(bm25_ss)),
               "std_ss": float(np.std(bm25_ss)), "nonzero_pct": float(np.mean([s > 0.001 for s in bm25_ss]))}
print(f"  n={bm25_result['n']}, mean_ss={bm25_result['mean_ss']:.6f}, nonzero={bm25_result['nonzero_pct']:.1%}")

# ═══════════════════════════════════════════════════════
# 2. SPLADE (sparse)
# ═══════════════════════════════════════════════════════
print("\n=== SPLADE ===")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    splade_id = "naver/splade-cocondenser-ensembledistil"
    splade_tok = AutoTokenizer.from_pretrained(splade_id)
    splade_model = AutoModelForMaskedLM.from_pretrained(splade_id)
    splade_model.eval()

    def splade_encode(text):
        inputs = splade_tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = splade_model(**inputs)
        # SPLADE: log(1 + ReLU(logits)) max-pooled over sequence
        rep = torch.log1p(torch.relu(out.logits)).max(dim=1).values.squeeze()
        return rep

    def splade_score(q_text, d_text):
        q_rep = splade_encode(q_text)
        d_rep = splade_encode(d_text)
        return (q_rep * d_rep).sum().item()

    splade_ss = []
    for qi, q in enumerate(QUERIES[:33]):
        for na, nb in NAME_PAIRS:
            for ti, t in enumerate(TEMPLATES):
                doc_a = t.format(name=na)
                doc_b = t.format(name=nb)
                sa = splade_score(q, doc_a)
                sb = splade_score(q, doc_b)
                avg = (sa + sb) / 2
                ss = abs(sa - sb) / avg if avg > 0 else 0
                splade_ss.append(ss)
        if (qi + 1) % 10 == 0:
            print(f"  SPLADE: {qi+1}/33 queries done")

    splade_result = {"model": "SPLADE", "n": len(splade_ss), "mean_ss": float(np.mean(splade_ss)),
                     "std_ss": float(np.std(splade_ss)), "median_ss": float(np.median(splade_ss))}
    print(f"  n={splade_result['n']}, mean_ss={splade_result['mean_ss']:.4f}")
    del splade_model, splade_tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  SPLADE failed: {e}")
    splade_result = None

# ═══════════════════════════════════════════════════════
# 3. Cross-Encoder
# ═══════════════════════════════════════════════════════
print("\n=== Cross-Encoder ===")
try:
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    ce_ss = []
    for qi, q in enumerate(QUERIES[:33]):
        for na, nb in NAME_PAIRS:
            for ti, t in enumerate(TEMPLATES):
                doc_a = t.format(name=na)
                doc_b = t.format(name=nb)
                scores = ce_model.predict([(q, doc_a), (q, doc_b)])
                sa, sb = float(scores[0]), float(scores[1])
                avg = (sa + sb) / 2
                ss = abs(sa - sb) / abs(avg) if abs(avg) > 1e-8 else 0
                ce_ss.append(ss)
        if (qi + 1) % 10 == 0:
            print(f"  CE: {qi+1}/33 queries done")

    ce_result = {"model": "CrossEncoder", "n": len(ce_ss), "mean_ss": float(np.mean(ce_ss)),
                 "std_ss": float(np.std(ce_ss)), "median_ss": float(np.median(ce_ss))}
    print(f"  n={ce_result['n']}, mean_ss={ce_result['mean_ss']:.6f}")
    del ce_model
except Exception as e:
    print(f"  CrossEncoder failed: {e}")
    ce_result = None

# ═══════════════════════════════════════════════════════
# 4. DPR (using sentence-transformers)
# ═══════════════════════════════════════════════════════
print("\n=== DPR (via sentence-transformers) ===")
try:
    from sentence_transformers import SentenceTransformer
    dpr_model = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")

    # DPR has separate query/passage encoders; use ctx encoder for both as proxy
    dpr_ss = []
    for qi, q in enumerate(QUERIES[:33]):
        for na, nb in NAME_PAIRS:
            for ti, t in enumerate(TEMPLATES):
                doc_a = t.format(name=na)
                doc_b = t.format(name=nb)
                embs = dpr_model.encode([q, doc_a, doc_b])
                sa = float(np.dot(embs[0], embs[1]))
                sb = float(np.dot(embs[0], embs[2]))
                avg = (sa + sb) / 2
                ss = abs(sa - sb) / abs(avg) if abs(avg) > 1e-8 else 0
                dpr_ss.append(ss)
        if (qi + 1) % 10 == 0:
            print(f"  DPR: {qi+1}/33 queries done")

    dpr_result = {"model": "DPR", "n": len(dpr_ss), "mean_ss": float(np.mean(dpr_ss)),
                  "std_ss": float(np.std(dpr_ss)), "median_ss": float(np.median(dpr_ss))}
    print(f"  n={dpr_result['n']}, mean_ss={dpr_result['mean_ss']:.4f}")
    del dpr_model
except Exception as e:
    print(f"  DPR failed: {e}")
    dpr_result = None

# ═══════════════════════════════════════════════════════
# 5. Contriever
# ═══════════════════════════════════════════════════════
print("\n=== Contriever ===")
try:
    from transformers import AutoTokenizer as AT2, AutoModel
    ct_tok = AT2.from_pretrained("facebook/contriever")
    ct_model = AutoModel.from_pretrained("facebook/contriever")
    ct_model.eval()

    def contriever_encode(text):
        inputs = ct_tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = ct_model(**inputs)
        # Mean pooling
        emb = out.last_hidden_state.mean(dim=1).squeeze()
        return emb.numpy()

    ct_ss = []
    for qi, q in enumerate(QUERIES[:33]):
        for na, nb in NAME_PAIRS:
            for ti, t in enumerate(TEMPLATES):
                doc_a = t.format(name=na)
                doc_b = t.format(name=nb)
                q_emb = contriever_encode(q)
                a_emb = contriever_encode(doc_a)
                b_emb = contriever_encode(doc_b)
                sa = float(np.dot(q_emb, a_emb))
                sb = float(np.dot(q_emb, b_emb))
                avg = (sa + sb) / 2
                ss = abs(sa - sb) / abs(avg) if abs(avg) > 1e-8 else 0
                ct_ss.append(ss)
        if (qi + 1) % 10 == 0:
            print(f"  Contriever: {qi+1}/33 queries done")

    ct_result = {"model": "Contriever", "n": len(ct_ss), "mean_ss": float(np.mean(ct_ss)),
                 "std_ss": float(np.std(ct_ss)), "median_ss": float(np.median(ct_ss))}
    print(f"  n={ct_result['n']}, mean_ss={ct_result['mean_ss']:.4f}")
    del ct_model, ct_tok
except Exception as e:
    print(f"  Contriever failed: {e}")
    ct_result = None

# ═══════════════════════════════════════════════════════
# Save all results
# ═══════════════════════════════════════════════════════
all_results = {"bm25": bm25_result}
if splade_result: all_results["splade"] = splade_result
if ce_result: all_results["cross_encoder"] = ce_result
if dpr_result: all_results["dpr"] = dpr_result
if ct_result: all_results["contriever"] = ct_result

with open(RESULTS / "multi_model_scaled.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n=== SUMMARY ===")
for k, v in all_results.items():
    print(f"  {k}: n={v['n']}, mean_ss={v['mean_ss']:.6f}")
print(f"\nSaved to {RESULTS / 'multi_model_scaled.json'}")
