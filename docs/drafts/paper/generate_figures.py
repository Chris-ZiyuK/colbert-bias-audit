#!/usr/bin/env python3
"""
Generate all figures for the ColBERT Bias Audit paper.
Usage: cd paper && python generate_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

# ── Design System ──────────────────────────────────────────────────────────

COLORS = {
    'function':   '#2C73D2',
    'content':    '#FF9F43',
    'accent':     '#E63946',
    'neutral':    '#6C757D',
    'bm25':       '#ADB5BD',
    'sig_star':   '#2D6A4F',
    'light_blue': '#A8D5F2',
    'pronoun':    '#845EC2',
    'bg_band':    '#F7F7F7',
}

COLUMN_WIDTH = 3.33   # inches, ACM sigconf single column
DOUBLE_WIDTH = 7.0    # inches, ACM sigconf double column

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })

def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'))
    fig.savefig(os.path.join(out_dir, f'{name}.png'))
    print(f'  ✓ saved figures/{name}.pdf + .png')
    plt.close(fig)

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── Fig 1: TCD Method Schematic ───────────────────────────────────────────

def fig1_tcd_schematic():
    """Conceptual diagram showing TCD computation."""
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH, 2.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # ── Query tokens ──
    query_tokens = ['Who', 'is', 'a', 'qualified', 'doctor', '?']
    token_colors = [COLORS['function']] * 3 + [COLORS['content']] * 2 + [COLORS['function']]
    token_labels = ['F', 'F', 'F', 'C', 'C', 'F']

    ax.text(0.3, 3.55, 'Query Q', fontsize=9, fontweight='bold', va='center')
    for i, (tok, col, lab) in enumerate(zip(query_tokens, token_colors, token_labels)):
        x = 1.8 + i * 1.1
        box = FancyBboxPatch((x - 0.4, 3.2), 0.8, 0.7,
                             boxstyle="round,pad=0.08", linewidth=0.8,
                             edgecolor=col, facecolor=col, alpha=0.15)
        ax.add_patch(box)
        ax.text(x, 3.55, tok, ha='center', va='center', fontsize=8,
                fontweight='bold', color=col)
        ax.text(x, 3.1, lab, ha='center', va='center', fontsize=6.5,
                color=col, fontstyle='italic')

    # ── Two documents ──
    doc_y = [2.0, 0.7]
    doc_names = ['Emily', 'Lakisha']
    doc_labels = ['$D_A$: "...Emily is a qualified doctor..."',
                  '$D_B$: "...Lakisha is a qualified doctor..."']
    for j, (y, name, label) in enumerate(zip(doc_y, doc_names, doc_labels)):
        box = FancyBboxPatch((0.2, y - 0.25), 5.0, 0.5,
                             boxstyle="round,pad=0.1", linewidth=0.8,
                             edgecolor=COLORS['neutral'], facecolor='white')
        ax.add_patch(box)
        ax.text(2.7, y, label, ha='center', va='center', fontsize=7.5)

    # ── MaxSim arrows ──
    ax.annotate('', xy=(5.6, 2.0), xytext=(3.5, 3.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'],
                                lw=0.8, connectionstyle='arc3,rad=-0.15'))
    ax.annotate('', xy=(5.6, 0.7), xytext=(3.5, 3.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'],
                                lw=0.8, connectionstyle='arc3,rad=0.15'))
    ax.text(5.9, 2.5, 'MaxSim', fontsize=7, color=COLORS['neutral'],
            fontstyle='italic', rotation=-20)
    ax.text(5.9, 1.0, 'MaxSim', fontsize=7, color=COLORS['neutral'],
            fontstyle='italic', rotation=20)

    # ── Per-token contributions ──
    c_a = [0.82, 0.91, 0.78, 0.95, 0.97, 0.73]
    c_b = [0.82, 0.87, 0.75, 0.95, 0.97, 0.73]
    tcd_vals = [abs(a - b) for a, b in zip(c_a, c_b)]

    contrib_x = 7.5
    ax.text(contrib_x - 0.3, 3.55, 'Per-token contributions', fontsize=8,
            fontweight='bold')

    headers = ['Token', '$c_i^A$', '$c_i^B$', 'TCD']
    for k, h in enumerate(headers):
        ax.text(contrib_x + k * 1.1, 3.1, h, ha='center', va='center',
                fontsize=7, fontweight='bold', color=COLORS['neutral'])

    for i, (tok, ca, cb, tcd) in enumerate(zip(query_tokens, c_a, c_b, tcd_vals)):
        y = 2.7 - i * 0.35
        col = token_colors[i]
        bg_alpha = 0.08 if i % 2 == 0 else 0.0

        if bg_alpha > 0:
            rect = plt.Rectangle((contrib_x - 0.6, y - 0.15), 4.5, 0.3,
                                  facecolor=COLORS['bg_band'], edgecolor='none',
                                  zorder=0)
            ax.add_patch(rect)

        ax.text(contrib_x, y, tok, ha='center', va='center', fontsize=7.5,
                color=col, fontweight='bold')
        ax.text(contrib_x + 1.1, y, f'{ca:.2f}', ha='center', va='center',
                fontsize=7.5)
        ax.text(contrib_x + 2.2, y, f'{cb:.2f}', ha='center', va='center',
                fontsize=7.5)

        tcd_color = COLORS['accent'] if tcd > 0.02 else COLORS['neutral']
        ax.text(contrib_x + 3.3, y, f'{tcd:.3f}', ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=tcd_color)

    # ── Key insight callout ──
    box = FancyBboxPatch((11.8, 0.3), 2.0, 1.2,
                         boxstyle="round,pad=0.12", linewidth=1.2,
                         edgecolor=COLORS['function'], facecolor=COLORS['function'],
                         alpha=0.08)
    ax.add_patch(box)
    ax.text(12.8, 1.15, 'Key insight:', fontsize=7, fontweight='bold',
            ha='center', color=COLORS['function'])
    ax.text(12.8, 0.8, 'Function words\n(is, a) show\nhigher TCD', fontsize=6.5,
            ha='center', va='center', color=COLORS['function'])

    # ── Legend ──
    func_patch = mpatches.Patch(facecolor=COLORS['function'], alpha=0.3,
                                 edgecolor=COLORS['function'], label='Function word')
    cont_patch = mpatches.Patch(facecolor=COLORS['content'], alpha=0.3,
                                 edgecolor=COLORS['content'], label='Content word')
    ax.legend(handles=[func_patch, cont_patch], loc='lower left',
              frameon=True, framealpha=0.9, fontsize=7)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    save_fig(fig, 'tcd_schematic')


# ── Fig 2: Per-Profession Func vs Content ─────────────────────────────────

def fig2_func_vs_cont_professions():
    """Horizontal paired bar chart of 33 professions."""
    csv_path = os.path.join(RESULTS, 'p1_profession_expansion', 'profession_summary.csv')
    df = pd.read_csv(csv_path)
    df = df.sort_values('mean_func_tcd', ascending=True)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.8))
    y = np.arange(len(df))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, df['mean_func_tcd'], bar_h,
            color=COLORS['function'], alpha=0.85, label='Func-TCD', zorder=3)
    ax.barh(y - bar_h / 2, df['mean_cont_tcd'], bar_h,
            color=COLORS['content'], alpha=0.85, label='Cont-TCD', zorder=3)

    # Mark func > cont
    for i, (_, row) in enumerate(df.iterrows()):
        if row['mean_func_tcd'] > row['mean_cont_tcd']:
            ax.text(max(row['mean_func_tcd'], row['mean_cont_tcd']) + 0.0005,
                    y[i], '●', fontsize=5, color=COLORS['sig_star'],
                    va='center', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([p.replace('_', ' ').title() for p in df['profession']],
                       fontsize=6)
    ax.set_xlabel('Mean TCD')
    ax.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.9)
    ax.set_title('Function vs Content Word TCD\nAcross 33 Professions', fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'func_vs_cont_professions')


# ── Fig 3: Name vs Pronoun Swap ───────────────────────────────────────────

def fig3_name_vs_pronoun():
    """Grouped bar chart comparing name and pronoun swaps."""
    json_path = os.path.join(RESULTS, 'p1_profession_expansion', 'p1_wrapup_summary.json')
    with open(json_path) as f:
        data = json.load(f)

    swap_data = {s['swap_type']: s for s in data['swap_type_summary']}
    metrics = ['mean_ss', 'mean_func_tcd', 'mean_func_minus_cont']
    labels = ['Score\nSensitivity', 'Func-TCD', 'Func−Cont\nGap']
    sig_markers = ['n.s.', '**', '*']  # from p1_wrapup_stats

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))

    name_vals = [swap_data['name'][m] for m in metrics]
    pron_vals = [swap_data['pronoun'][m] for m in metrics]

    bars1 = ax.bar(x - width/2, name_vals, width, color=COLORS['function'],
                   alpha=0.85, label='Name swap', zorder=3)
    bars2 = ax.bar(x + width/2, pron_vals, width, color=COLORS['pronoun'],
                   alpha=0.85, label='Pronoun swap', zorder=3)

    # Significance annotations
    for i, sig in enumerate(sig_markers):
        if sig != 'n.s.':
            max_h = max(name_vals[i], pron_vals[i])
            ax.text(x[i], max_h + 0.001, sig, ha='center', fontsize=7,
                    color=COLORS['accent'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Value')
    ax.legend(loc='upper left', fontsize=7, frameon=True, framealpha=0.9)
    ax.set_title('Name vs. Pronoun Swap Effects', fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'name_vs_pronoun')


# ── Fig 4: Rarity Amplification ──────────────────────────────────────────

def fig4_rarity_amplification():
    """Grouped bar chart of rarity categories."""
    categories = ['Common–\nCommon', 'Common–\nRare', 'Rare–\nRare']
    ss_vals = [0.0210, 0.0305, 0.0366]
    func_vals = [0.0108, 0.0163, 0.0199]
    ratios = ['1.0×', '1.5×', '1.84×']

    x = np.arange(len(categories))
    width = 0.3

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))

    bars1 = ax.bar(x - width/2, ss_vals, width, color=COLORS['light_blue'],
                   alpha=0.9, label='Score Sensitivity', zorder=3,
                   edgecolor=COLORS['function'], linewidth=0.5)
    bars2 = ax.bar(x + width/2, func_vals, width, color=COLORS['function'],
                   alpha=0.85, label='Func-TCD', zorder=3)

    # Ratio annotations
    for i, (ratio, ss, func) in enumerate(zip(ratios, ss_vals, func_vals)):
        max_h = max(ss, func)
        weight = 'bold' if i == 2 else 'normal'
        color = COLORS['accent'] if i == 2 else COLORS['neutral']
        ax.text(x[i], max_h + 0.002, ratio, ha='center', fontsize=8,
                fontweight=weight, color=color)

    # Significance markers for last group
    ax.text(x[2], ss_vals[2] + 0.006, '***', ha='center', fontsize=7,
            color=COLORS['accent'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Value')
    ax.set_ylim(0, 0.048)
    ax.legend(loc='upper left', fontsize=7, frameon=True, framealpha=0.9)
    ax.set_title('Rarity Amplification of Bias', fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'rarity_amplification')


# ── Fig 5: Confound Decomposition Forest Plot ────────────────────────────

def fig5_decomposition_forest():
    """Forest plot of P2b confound decomposition coefficients."""
    json_path = os.path.join(RESULTS, 'p2_name_confound', 'p2b_rosenman_summary.json')
    with open(json_path) as f:
        data = json.load(f)

    factors = ['absolute_rarity', 'frequency_gap', 'race', 'tokenization', 'gender']
    labels = ['Absolute rarity', 'Frequency gap', 'Race', 'Tokenization', 'Gender']

    coefs, cis_lo, cis_hi, pvals, sigs = [], [], [], [], []
    for factor in factors:
        d = data[factor]['models']['func_tcd']
        coef = d['coef']
        se = d['stderr']
        coefs.append(coef)
        cis_lo.append(coef - 1.96 * se)
        cis_hi.append(coef + 1.96 * se)
        pvals.append(d['pvalue'])
        sigs.append(d['significant'])

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    y = np.arange(len(factors))

    for i in range(len(factors)):
        color = COLORS['function'] if sigs[i] else COLORS['neutral']
        marker = 'o' if sigs[i] else 'o'
        fillstyle = 'full' if sigs[i] else 'none'
        lw = 1.5 if sigs[i] else 0.8

        ax.plot([cis_lo[i], cis_hi[i]], [y[i], y[i]], color=color,
                linewidth=lw, zorder=3)
        ax.plot(coefs[i], y[i], marker=marker, color=color, markersize=6,
                fillstyle=fillstyle, markeredgewidth=1.2, zorder=4)

        # p-value label
        p = pvals[i]
        if p < 0.001:
            p_str = f'p < 0.001'
        elif p < 0.01:
            p_str = f'p = {p:.3f}'
        else:
            p_str = f'p = {p:.3f}'
        fw = 'bold' if sigs[i] else 'normal'
        ax.text(max(cis_hi[i], 0.006) + 0.0004, y[i], p_str,
                va='center', fontsize=6.5, color=color, fontweight=fw)

    ax.axvline(x=0, color=COLORS['neutral'], linestyle='--',
               linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Func-TCD Coefficient', fontsize=9)
    ax.set_title('Confound Decomposition\n(P2b Rosenman Validation)', fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'decomposition_forest')


# ── Fig 6: R² Model Ladder ──────────────────────────────────────────────

def fig6_model_ladder():
    """Step plot showing R² improvement across nested models."""
    json_path = os.path.join(RESULTS, 'p2_name_confound', 'p2_final_summary.json')
    with open(json_path) as f:
        data = json.load(f)

    model_names = ['Race\nonly', '+ Gender', '+ Token\n/ Freq', '+ Full\n(clust. SE)']
    ss_r2 = [m['r_squared'] for m in data['nested_models']['ss']]
    func_r2 = [m['r_squared'] for m in data['nested_models']['func_tcd']]

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))

    # Lines
    ax.plot(x, func_r2, 'o-', color=COLORS['function'], markersize=7,
            linewidth=1.8, label='Func-TCD $R^2$', zorder=4)
    ax.plot(x, ss_r2, 's-', color=COLORS['light_blue'], markersize=6,
            linewidth=1.5, label='SS $R^2$', zorder=3,
            markeredgecolor=COLORS['function'], markeredgewidth=0.5)

    # ΔR² annotations
    for i in range(1, len(func_r2)):
        delta = func_r2[i] - func_r2[i - 1]
        mid_y = (func_r2[i] + func_r2[i - 1]) / 2
        color = COLORS['accent'] if delta > 0.02 else COLORS['neutral']
        ax.annotate(f'Δ={delta:.3f}', xy=(x[i] - 0.5, mid_y),
                    fontsize=6.5, color=color, ha='center',
                    fontweight='bold' if delta > 0.02 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=7.5)
    ax.set_ylabel('$R^2$')
    ax.set_ylim(0.14, 0.215)
    ax.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.9)
    ax.set_title('Nested Model $R^2$ Progression', fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'model_ladder')


# ── Fig 7: Cross-Setting Validation (Two Panels) ────────────────────────

def fig7_cross_setting_validation():
    """Two-panel figure: (A) 3-setting comparison, (B) per-category breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.8),
                                    gridspec_kw={'width_ratios': [1, 1.3]})

    # ── Panel A: Three-setting comparison ──
    settings = ['Synthetic\ntemplates', 'Natural\ntext', 'BM25\nbaseline']
    func_vals = [0.0177, 0.0124, 0.0000]
    cont_vals = [0.0133, 0.0087, 0.0000]
    ratios = ['1.44×', '1.42×', '—']

    x = np.arange(len(settings))
    width = 0.3

    colors_func = [COLORS['function'], COLORS['function'], COLORS['bm25']]
    colors_cont = [COLORS['content'], COLORS['content'], COLORS['bm25']]

    for i in range(len(settings)):
        ax1.bar(x[i] - width/2, func_vals[i], width, color=colors_func[i],
                alpha=0.85, zorder=3)
        ax1.bar(x[i] + width/2, cont_vals[i], width, color=colors_cont[i],
                alpha=0.85, zorder=3)

        max_h = max(func_vals[i], cont_vals[i])
        if max_h > 0:
            ax1.text(x[i], max_h + 0.001, ratios[i], ha='center', fontsize=7.5,
                     fontweight='bold', color=COLORS['accent'] if i < 2 else COLORS['neutral'])
        else:
            ax1.text(x[i], 0.001, '0.00',  ha='center', fontsize=7,
                     color=COLORS['neutral'])

    # Dashed line connecting the two ColBERT ratios
    ax1.plot([0, 1], [0.019, 0.0145], '--', color=COLORS['accent'],
             linewidth=0.7, alpha=0.5)
    ax1.text(0.5, 0.0175, '≈ identical\nratio', fontsize=5.5, ha='center',
             color=COLORS['accent'], fontstyle='italic')

    func_p = mpatches.Patch(color=COLORS['function'], alpha=0.85, label='Func-TCD')
    cont_p = mpatches.Patch(color=COLORS['content'], alpha=0.85, label='Cont-TCD')
    bm25_p = mpatches.Patch(color=COLORS['bm25'], alpha=0.85, label='BM25 (0)')
    ax1.legend(handles=[func_p, cont_p, bm25_p], fontsize=6, loc='upper right',
               frameon=True, framealpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(settings, fontsize=7.5)
    ax1.set_ylabel('Mean TCD')
    ax1.set_ylim(0, 0.023)
    ax1.set_title('(A) Cross-Setting Comparison', fontsize=9)

    # ── Panel B: Per-category breakdown ──
    categories = ['community', 'education', 'evaluation', 'legal',
                  'medical', 'news', 'professional']
    func_cat = [0.015288, 0.016267, 0.008901, 0.007867,
                0.010196, 0.014667, 0.012345]
    cont_cat = [0.013709, 0.010939, 0.008714, 0.005052,
                0.005911, 0.008628, 0.008535]

    # Sort by func_tcd
    order = np.argsort(func_cat)
    categories = [categories[i] for i in order]
    func_cat = [func_cat[i] for i in order]
    cont_cat = [cont_cat[i] for i in order]

    y2 = np.arange(len(categories))
    bar_h = 0.35

    ax2.barh(y2 + bar_h/2, func_cat, bar_h, color=COLORS['function'],
             alpha=0.85, label='Func-TCD', zorder=3)
    ax2.barh(y2 - bar_h/2, cont_cat, bar_h, color=COLORS['content'],
             alpha=0.85, label='Cont-TCD', zorder=3)

    # Mark func > cont
    for i in range(len(categories)):
        if func_cat[i] > cont_cat[i]:
            ax2.text(max(func_cat[i], cont_cat[i]) + 0.0003, y2[i],
                     '●', fontsize=5, color=COLORS['sig_star'],
                     va='center', fontweight='bold')

    ax2.set_yticks(y2)
    ax2.set_yticklabels([c.title() for c in categories], fontsize=7.5)
    ax2.set_xlabel('Mean TCD')
    ax2.set_title('(B) Func vs Cont by Text Category', fontsize=9)
    ax2.legend(fontsize=6, loc='lower right', frameon=True, framealpha=0.9)

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'cross_setting_validation')


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    setup_style()
    print('Generating paper figures...\n')

    print('Fig 1: TCD Method Schematic')
    fig1_tcd_schematic()

    print('Fig 2: Per-Profession Func vs Content')
    fig2_func_vs_cont_professions()

    print('Fig 3: Name vs Pronoun Swap')
    fig3_name_vs_pronoun()

    print('Fig 4: Rarity Amplification')
    fig4_rarity_amplification()

    print('Fig 5: Confound Decomposition Forest Plot')
    fig5_decomposition_forest()

    print('Fig 6: R² Model Ladder')
    fig6_model_ladder()

    print('Fig 7: Cross-Setting Validation')
    fig7_cross_setting_validation()

    print('\n✅ All 7 figures generated successfully!')


if __name__ == '__main__':
    main()
