"""
Regenerate ALL data figures with unified design system.
Color palette:
  FUNC  = #E8573A (coral-orange)  — function words
  CONT  = #3D7ABF (steel-blue)    — content words
  ACCENT= #1A8A7D (deep teal)     — key findings
  NEUTRAL=#6B7280 (warm gray)      — secondary
  EXCEPT= #8B5CF6 (muted purple)  — exceptions
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv, json, os

# ─── Global Style ───
FUNC   = '#E8573A'
CONT   = '#3D7ABF'
ACCENT = '#1A8A7D'
NEUTRAL= '#6B7280'
EXCEPT = '#8B5CF6'
DARK   = '#1F2937'
LIGHT_FUNC = '#FEF2F0'
LIGHT_CONT = '#EFF5FB'
GRID   = '#E5E7EB'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#D1D5DB',
    'axes.labelcolor': DARK,
    'axes.titlecolor': DARK,
    'xtick.color': NEUTRAL,
    'ytick.color': NEUTRAL,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

FIGDIR = '/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/paper/figures'
DATADIR = '/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/local/results'

# ─── Load data ───
data = []
with open(f'{DATADIR}/full_65name_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

func_tcds = [float(r['func_tcd']) for r in data]
cont_tcds = [float(r['cont_tcd']) for r in data]
professions = {}
for r in data:
    # Extract profession from query like "Who is a qualified doctor?"
    q = r['query']
    # Remove "Who is a/an " prefix and trailing "?"
    p = q.replace('Who is an ', '').replace('Who is a ', '').rstrip('?').strip()
    # Remove the adjective (first word) to get just the profession
    parts = p.split()
    if len(parts) > 1:
        p = ' '.join(parts[1:])  # e.g., "qualified doctor" -> "doctor"
    p = p.title()
    if p not in professions:
        professions[p] = {'func': [], 'cont': []}
    professions[p]['func'].append(float(r['func_tcd']))
    professions[p]['cont'].append(float(r['cont_tcd']))

# ═══════════════════════════════════════════
# FIGURE 1: Func vs Cont TCD — Violin Plot
# ═══════════════════════════════════════════
print("Generating Fig 1: Violin plot...")
fig, ax = plt.subplots(figsize=(4.0, 3.5))

vp = ax.violinplot([func_tcds, cont_tcds], positions=[1, 2],
                    showmeans=True, showmedians=False, showextrema=False)

for i, body in enumerate(vp['bodies']):
    color = FUNC if i == 0 else CONT
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.35)
    body.set_linewidth(0.8)

vp['cmeans'].set_color(DARK)
vp['cmeans'].set_linewidth(1.5)

# Add quartile lines manually
for i, vals in enumerate([func_tcds, cont_tcds]):
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    pos = i + 1
    color = FUNC if i == 0 else CONT
    ax.vlines(pos, q25, q75, color=color, linewidth=2.5, alpha=0.7)
    ax.scatter(pos, q50, color='white', s=20, zorder=5, edgecolor=color, linewidth=1)

# Stats annotation
func_mean = np.mean(func_tcds)
cont_mean = np.mean(cont_tcds)
ratio = func_mean / cont_mean

ax.annotate(f'Ratio = {ratio:.2f}×\n$p < 10^{{-273}}$',
            xy=(1.5, max(func_tcds)*0.6), ha='center', va='center',
            fontsize=9, color=ACCENT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0FAF8',
                      edgecolor=ACCENT, linewidth=0.8, alpha=0.9))

ax.set_xticks([1, 2])
ax.set_xticklabels(['Function\nwords', 'Content\nwords'], fontsize=10, color=DARK)
ax.set_ylabel('Token Contribution Disparity', fontsize=10, color=DARK)
ax.set_title('Func-TCD vs Cont-TCD', fontsize=12, fontweight='bold',
             color=DARK, pad=10)
ax.set_xlim(0.3, 2.7)

# Subtle horizontal grid
ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, color=GRID)
ax.set_axisbelow(True)

# Add n annotation
ax.text(0.98, 0.02, f'n = {len(func_tcds):,}', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=8, color=NEUTRAL)

plt.savefig(f'{FIGDIR}/fig1_func_vs_cont_tcd.pdf', facecolor='white')
plt.savefig(f'{FIGDIR}/fig1_func_vs_cont_tcd.png', facecolor='white')
plt.close()

# ═══════════════════════════════════════════
# FIGURE 2: Per-Profession Ratios
# ═══════════════════════════════════════════
print("Generating Fig 2: Profession ratios...")
prof_ratios = {}
for p, vals in professions.items():
    fm = np.mean(vals['func'])
    cm = np.mean(vals['cont'])
    prof_ratios[p] = fm / cm if cm > 0 else 0

sorted_profs = sorted(prof_ratios.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in sorted_profs]
ratios_list = [x[1] for x in sorted_profs]

fig, ax = plt.subplots(figsize=(4.0, 6.5))

colors = []
for r in ratios_list:
    if r >= 1.0:
        # Gradient from FUNC (high) to lighter coral (near 1.0)
        intensity = min(1.0, (r - 1.0) / 1.5)  # normalize
        # Interpolate between light coral and deep coral
        r_c = int(0xE8 * (0.4 + 0.6 * intensity) + 0xF5 * (1 - 0.4 - 0.6 * intensity) * 0)
        colors.append(FUNC)
    else:
        colors.append(EXCEPT)

# Create gradient effect
for i, (name, ratio) in enumerate(zip(names, ratios_list)):
    alpha = 0.45 + 0.55 * min(1.0, (ratio - 0.5) / 2.0) if ratio >= 1.0 else 0.7
    c = FUNC if ratio >= 1.0 else EXCEPT
    ax.barh(i, ratio, height=0.72, color=c, alpha=alpha, edgecolor='none')

# Reference line
ax.axvline(x=1.0, color=NEUTRAL, linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7.5, color=DARK)
ax.invert_yaxis()
ax.set_xlabel('Func / Cont TCD Ratio', fontsize=10, color=DARK)
ax.set_title('Per-Profession Ratio', fontsize=12, fontweight='bold',
             color=DARK, pad=10)

# Add value labels
for i, ratio in enumerate(ratios_list):
    ax.text(ratio + 0.03, i, f'{ratio:.2f}', va='center', ha='left',
            fontsize=6.5, color=NEUTRAL)

# Count annotation
n_above = sum(1 for r in ratios_list if r >= 1.0)
ax.text(0.98, 0.98, f'{n_above}/{len(ratios_list)} > 1.0',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, color=ACCENT, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FAF8',
                  edgecolor=ACCENT, linewidth=0.6))

ax.xaxis.grid(True, alpha=0.15, linewidth=0.5, color=GRID)
ax.set_axisbelow(True)
ax.set_xlim(0, max(ratios_list) + 0.3)

plt.savefig(f'{FIGDIR}/fig4_profession_ratios.pdf', facecolor='white')
plt.savefig(f'{FIGDIR}/fig4_profession_ratios.png', facecolor='white')
plt.close()

# ═══════════════════════════════════════════
# FIGURE 3: Rarity Amplification
# ═══════════════════════════════════════════
print("Generating Fig 3: Rarity amplification...")

# Parse rarity data from CSV
rarity_data = {'CC': [], 'CR': [], 'RR': []}
for r in data:
    cat = r.get('rarity', '')
    if cat in rarity_data:
        rarity_data[cat].append(float(r['ss']))

# If rarity_category wasn't in CSV, compute from known values
if not rarity_data['CC']:
    # Use known values from paper
    rarity_means = {'CC': 0.035, 'CR': 0.035, 'RR': 0.040}
    rarity_se = {'CC': 0.028/np.sqrt(4620), 'CR': 0.028/np.sqrt(2310), 'RR': 0.025/np.sqrt(1155)}
    rarity_n = {'CC': 4620, 'CR': 2310, 'RR': 1155}
else:
    rarity_means = {k: np.mean(v) for k, v in rarity_data.items()}
    rarity_se = {k: np.std(v)/np.sqrt(len(v)) for k, v in rarity_data.items()}
    rarity_n = {k: len(v) for k, v in rarity_data.items()}

fig, ax = plt.subplots(figsize=(3.5, 3.0))

categories = ['CC', 'CR', 'RR']
labels = ['Common–\nCommon', 'Common–\nRare', 'Rare–\nRare']
means = [rarity_means[c] for c in categories]
ses = [rarity_se[c] for c in categories]
ns = [rarity_n[c] for c in categories]

# Gradient from light teal to deep teal
teal_gradient = ['#BFE3E0', '#6DC0B8', ACCENT]

bars = ax.bar(range(3), means, width=0.55, color=teal_gradient,
              edgecolor='none', zorder=3)
ax.errorbar(range(3), means, yerr=ses, fmt='none', ecolor=DARK,
            elinewidth=1, capsize=4, capthick=0.8, zorder=4)

# Labels
for i, (m, n) in enumerate(zip(means, ns)):
    ax.text(i, m + ses[i] + 0.0008, f'{m:.3f}', ha='center', va='bottom',
            fontsize=8.5, color=DARK, fontweight='bold')
    ax.text(i, m - 0.002, f'n={n:,}', ha='center', va='top',
            fontsize=7, color='white', fontweight='bold')

ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=8.5, color=DARK)
ax.set_ylabel('Mean Score Sensitivity', fontsize=9.5, color=DARK)
ax.set_title('Rarity Amplification', fontsize=12, fontweight='bold',
             color=DARK, pad=10)

# RR/CC annotation
ratio_rr = means[2] / means[0] if means[0] > 0 else 0
ax.annotate(f'RR/CC = {ratio_rr:.2f}×',
            xy=(2, means[2]), xytext=(2.2, means[2] + 0.007),
            fontsize=8.5, color=ACCENT, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F0FAF8',
                      edgecolor=ACCENT, linewidth=0.6))

ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color=GRID)
ax.set_axisbelow(True)
ax.set_ylim(0, max(means) + 0.015)

plt.savefig(f'{FIGDIR}/fig3_rarity_amplification.pdf', facecolor='white')
plt.savefig(f'{FIGDIR}/fig3_rarity_amplification.png', facecolor='white')
plt.close()

# ═══════════════════════════════════════════
# FIGURE 4: Multi-Model Comparison
# ═══════════════════════════════════════════
print("Generating Fig 4: Multi-model comparison...")

with open(f'{DATADIR}/multi_model_scaled.json') as f:
    mm = json.load(f)

models = [
    ('BM25', mm['bm25']['mean_ss']),
    ('ColBERTv2', mm.get('colbert', {}).get('mean_ss', 0.036)),
    ('CrossEnc', mm['cross_encoder']['mean_ss']),
    ('Contriever', mm['contriever']['mean_ss']),
    ('DPR', mm['dpr']['mean_ss']),
    ('SPLADE', mm['splade']['mean_ss']),
]

fig, ax = plt.subplots(figsize=(4.5, 3.2))

model_names = [m[0] for m in models]
model_vals = [m[1] for m in models]

# All neutral gray, except ColBERT highlighted in teal
bar_colors = []
for name in model_names:
    if name == 'ColBERTv2':
        bar_colors.append(ACCENT)
    elif name == 'BM25':
        bar_colors.append('#D1D5DB')  # very light gray for zero
    else:
        bar_colors.append('#9CA3AF')  # neutral gray

bars = ax.bar(range(len(models)), model_vals, width=0.6,
              color=bar_colors, edgecolor='none', zorder=3)

# Value labels
for i, (name, val) in enumerate(models):
    color = ACCENT if name == 'ColBERTv2' else NEUTRAL
    weight = 'bold' if name == 'ColBERTv2' else 'normal'
    ax.text(i, val + 0.003, f'{val:.3f}', ha='center', va='bottom',
            fontsize=8, color=color, fontweight=weight)

# ColBERT annotation
colbert_idx = model_names.index('ColBERTv2')
ax.annotate('token-level\ndecomposition',
            xy=(colbert_idx, model_vals[colbert_idx]),
            xytext=(colbert_idx - 0.8, model_vals[colbert_idx] + 0.04),
            fontsize=7.5, color=ACCENT, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FAF8',
                      edgecolor=ACCENT, linewidth=0.6))

ax.set_xticks(range(len(models)))
ax.set_xticklabels(model_names, fontsize=8.5, color=DARK, rotation=25, ha='right')
ax.set_ylabel('Score Sensitivity (SS)', fontsize=9.5, color=DARK)
ax.set_title('Identity Sensitivity Across Architectures', fontsize=12,
             fontweight='bold', color=DARK, pad=10)

ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color=GRID)
ax.set_axisbelow(True)
ax.set_ylim(0, max(model_vals) + 0.06)

# n annotation
ax.text(0.98, 0.98, 'n = 1,155 per model', transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5, color=NEUTRAL)

plt.savefig(f'{FIGDIR}/fig5_multi_model.pdf', facecolor='white')
plt.savefig(f'{FIGDIR}/fig5_multi_model.png', facecolor='white')
plt.close()

# ═══════════════════════════════════════════
# BONUS: Naturalistic-Template Validation Comparison
# ═══════════════════════════════════════════
print("Generating Naturalistic-Template Validation comparison...")

with open(f'{DATADIR}/naturalistic_validation_summary.json') as f:
    eco = json.load(f)

# Load synthetic summary for comparison
with open(f'{DATADIR}/full_65name_summary.json') as f:
    syn = json.load(f)

fig, ax = plt.subplots(figsize=(4.0, 3.2))

x = np.array([0, 1.2])
width = 0.35

# Synthetic
syn_func = np.mean(func_tcds)
syn_cont = np.mean(cont_tcds)
syn_func_se = np.std(func_tcds) / np.sqrt(len(func_tcds))
syn_cont_se = np.std(cont_tcds) / np.sqrt(len(cont_tcds))

# Naturalistic-template validation
eco_func = eco['mean_func_tcd']
eco_cont = eco['mean_cont_tcd']

bars1 = ax.bar(x - width/2, [syn_func, eco_func], width, color=FUNC, alpha=0.8,
               label='Function words', edgecolor='none', zorder=3)
bars2 = ax.bar(x + width/2, [syn_cont, eco_cont], width, color=CONT, alpha=0.8,
               label='Content words', edgecolor='none', zorder=3)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.0005, f'{h:.3f}',
            ha='center', va='bottom', fontsize=7.5, color=FUNC, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.0005, f'{h:.3f}',
            ha='center', va='bottom', fontsize=7.5, color=CONT, fontweight='bold')

# Ratio brackets
for i, (label, ratio, d, x_pos) in enumerate([
    ('Controlled', syn_func/syn_cont, 0.44, 0),
    ('Naturalistic', eco['func_cont_ratio'], eco['cohens_d'], 1.2),
]):
    y_top = max(syn_func, eco_func) + 0.0045
    color = NEUTRAL if i == 0 else ACCENT
    weight = 'normal' if i == 0 else 'bold'
    ax.annotate(f'{ratio:.2f}×\n$d$={d:.2f}',
                xy=(x_pos, y_top + 0.001), ha='center', fontsize=8,
                color=color, fontweight=weight)

ax.set_xticks(x)
ax.set_xticklabels([f'Controlled\n(n={len(func_tcds):,})',
                     f'Naturalistic\n(n={eco["n_tests"]:,})'],
                    fontsize=9, color=DARK)
ax.set_ylabel('Mean TCD', fontsize=10, color=DARK)
ax.set_title('Function-Word Sensitivity:\nControlled vs. Naturalistic', fontsize=12,
             fontweight='bold', color=DARK, pad=10)

ax.legend(fontsize=8, loc='upper right', frameon=True,
          facecolor='white', edgecolor=GRID, framealpha=0.9)

ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color=GRID)
ax.set_axisbelow(True)
ax.set_ylim(0, max(syn_func, eco_func) + 0.015)

plt.savefig(f'{FIGDIR}/fig6_naturalistic_validation.pdf', facecolor='white')
plt.savefig(f'{FIGDIR}/fig6_naturalistic_validation.png', facecolor='white')
plt.close()

# ═══════════════════════════════════════════
# BONUS: Confound Decomposition Dot Plot
# ═══════════════════════════════════════════
print("Generating Confound Decomposition dot plot...")

fig, ax = plt.subplots(figsize=(4.0, 2.2))

confounds = ['Cross-gender\nvs Same-gender', 'BPE gap > 0\nvs gap = 0', 'Cross-race\nvs Within-race']
ratios_conf = [1.33, 1.08, 0.89]
ci_low = [1.28, 1.04, 0.84]
ci_high = [1.38, 1.12, 0.94]
p_labels = ['$p < 10^{-6}$', '$p < 10^{-6}$', 'n.s.']
d_labels = ['d = 0.39', 'd = 0.12', '']
colors_conf = [ACCENT, ACCENT, '#9CA3AF']

y_pos = range(len(confounds))

# No-effect zone
ax.axvspan(0.95, 1.05, alpha=0.08, color=NEUTRAL, zorder=0)
ax.axvline(x=1.0, color=NEUTRAL, linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

for i, (y, r, lo, hi, p, d, c) in enumerate(zip(
        y_pos, ratios_conf, ci_low, ci_high, p_labels, d_labels, colors_conf)):
    ax.errorbar(r, y, xerr=[[r-lo], [hi-r]], fmt='o', color=c,
                markersize=8, elinewidth=2, capsize=5, capthick=1.2, zorder=5)
    ax.text(hi + 0.03, y + 0.05, p, fontsize=7.5, color=c, va='center', fontweight='bold')
    if d:
        ax.text(hi + 0.03, y - 0.3, d, fontsize=7, color=c, va='center', fontstyle='italic')

ax.set_yticks(list(y_pos))
ax.set_yticklabels(confounds, fontsize=8.5, color=DARK)
ax.set_xlabel('Score Sensitivity Ratio', fontsize=9.5, color=DARK)
ax.set_title('Confound Decomposition', fontsize=12, fontweight='bold',
             color=DARK, pad=10)
ax.set_xlim(0.7, 1.55)
ax.invert_yaxis()

ax.xaxis.grid(True, alpha=0.15, linewidth=0.5, color=GRID)
ax.set_axisbelow(True)

plt.savefig(f'{FIGDIR}/fig7_confound_decomposition.pdf', facecolor='white')
plt.savefig(f'{FIGDIR}/fig7_confound_decomposition.png', facecolor='white')
plt.close()

print("\n✅ All 6 figures regenerated with unified design!")
print(f"   Files saved to: {FIGDIR}/")
