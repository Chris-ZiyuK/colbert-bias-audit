#!/usr/bin/env python3
"""
Regenerate only Figure 1 (TCD schematic) with corrected layout
to avoid element overlap in ACL two-column format.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

COLORS = {
    'function':   '#2C73D2',
    'content':    '#FF9F43',
    'accent':     '#E63946',
    'neutral':    '#6C757D',
    'bg_band':    '#F7F7F7',
}

ACL_DOUBLE_WIDTH = 6.3  # ACL \textwidth ≈ 160mm ≈ 6.3in

def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    # Use a taller figure to give vertical breathing room
    fig, ax = plt.subplots(figsize=(ACL_DOUBLE_WIDTH, 3.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.2, 4.5)
    ax.axis('off')

    # ── Query tokens (top row) ──
    query_tokens = ['Who', 'is', 'a', 'qualified', 'doctor', '?']
    token_colors = [COLORS['function']] * 3 + [COLORS['content']] * 2 + [COLORS['function']]
    token_labels = ['F', 'F', 'F', 'C', 'C', 'F']

    ax.text(0.2, 4.0, 'Query Q', fontsize=9, fontweight='bold', va='center')
    for i, (tok, col, lab) in enumerate(zip(query_tokens, token_colors, token_labels)):
        x = 1.8 + i * 1.05
        box = FancyBboxPatch((x - 0.42, 3.55), 0.84, 0.75,
                             boxstyle="round,pad=0.08", linewidth=0.8,
                             edgecolor=col, facecolor=col, alpha=0.15)
        ax.add_patch(box)
        ax.text(x, 3.93, tok, ha='center', va='center', fontsize=8,
                fontweight='bold', color=col)
        ax.text(x, 3.48, lab, ha='center', va='center', fontsize=6.5,
                color=col, fontstyle='italic')

    # ── Two documents (left side, stacked vertically) ──
    doc_labels = [r'$D_A$: "…Emily is a qualified doctor…"',
                  r'$D_B$: "…Lakisha is a qualified doctor…"']
    doc_ys = [2.35, 1.2]
    for y, label in zip(doc_ys, doc_labels):
        box = FancyBboxPatch((0.15, y - 0.25), 5.2, 0.5,
                             boxstyle="round,pad=0.1", linewidth=0.8,
                             edgecolor=COLORS['neutral'], facecolor='white')
        ax.add_patch(box)
        ax.text(2.75, y, label, ha='center', va='center', fontsize=7.5)

    # ── MaxSim arrows from query center to doc boxes ──
    q_center_x = 3.5
    ax.annotate('', xy=(5.5, 2.35), xytext=(q_center_x, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'],
                                lw=0.8, connectionstyle='arc3,rad=-0.15'))
    ax.annotate('', xy=(5.5, 1.2), xytext=(q_center_x, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'],
                                lw=0.8, connectionstyle='arc3,rad=0.15'))
    ax.text(5.75, 2.9, 'MaxSim', fontsize=7, color=COLORS['neutral'],
            fontstyle='italic', rotation=-25)
    ax.text(5.75, 1.65, 'MaxSim', fontsize=7, color=COLORS['neutral'],
            fontstyle='italic', rotation=25)

    # ── Per-token contribution table (right side) ──
    c_a = [0.82, 0.91, 0.78, 0.95, 0.97, 0.73]
    c_b = [0.82, 0.87, 0.75, 0.95, 0.97, 0.73]
    tcd_vals = [abs(a - b) for a, b in zip(c_a, c_b)]

    table_x = 8.0
    ax.text(table_x + 1.3, 4.0, 'Per-token contributions', fontsize=8,
            fontweight='bold', ha='center')

    headers = ['Token', r'$c_i^A$', r'$c_i^B$', 'TCD']
    col_offsets = [0, 1.0, 1.8, 2.6]
    for k, h in enumerate(headers):
        ax.text(table_x + col_offsets[k], 3.5, h, ha='center', va='center',
                fontsize=7, fontweight='bold', color=COLORS['neutral'])

    for i, (tok, ca, cb, tcd) in enumerate(zip(query_tokens, c_a, c_b, tcd_vals)):
        y = 3.05 - i * 0.38
        col = token_colors[i]

        if i % 2 == 0:
            rect = plt.Rectangle((table_x - 0.5, y - 0.15), 3.7, 0.3,
                                  facecolor=COLORS['bg_band'], edgecolor='none',
                                  zorder=0)
            ax.add_patch(rect)

        ax.text(table_x + col_offsets[0], y, tok, ha='center', va='center',
                fontsize=7.5, color=col, fontweight='bold')
        ax.text(table_x + col_offsets[1], y, f'{ca:.2f}', ha='center',
                va='center', fontsize=7.5)
        ax.text(table_x + col_offsets[2], y, f'{cb:.2f}', ha='center',
                va='center', fontsize=7.5)

        tcd_color = COLORS['accent'] if tcd > 0.02 else COLORS['neutral']
        ax.text(table_x + col_offsets[3], y, f'{tcd:.3f}', ha='center',
                va='center', fontsize=7.5, fontweight='bold', color=tcd_color)

    # ── Key insight callout (bottom-right, below table) ──
    box = FancyBboxPatch((table_x - 0.3, 0.15), 3.3, 0.75,
                         boxstyle="round,pad=0.12", linewidth=1.2,
                         edgecolor=COLORS['function'], facecolor=COLORS['function'],
                         alpha=0.08)
    ax.add_patch(box)
    ax.text(table_x + 1.35, 0.65, 'Key insight:', fontsize=7,
            fontweight='bold', ha='center', color=COLORS['function'])
    ax.text(table_x + 1.35, 0.35, 'Function words (is, a) show higher TCD',
            fontsize=6.5, ha='center', va='center', color=COLORS['function'])

    # ── Legend ──
    func_patch = mpatches.Patch(facecolor=COLORS['function'], alpha=0.3,
                                 edgecolor=COLORS['function'], label='Function word (F)')
    cont_patch = mpatches.Patch(facecolor=COLORS['content'], alpha=0.3,
                                 edgecolor=COLORS['content'], label='Content word (C)')
    ax.legend(handles=[func_patch, cont_patch], loc='lower left',
              frameon=True, framealpha=0.9, fontsize=7,
              bbox_to_anchor=(0.0, -0.02))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'tcd_schematic.pdf'))
    fig.savefig(os.path.join(out_dir, 'tcd_schematic.png'))
    print('✓ Regenerated figures/tcd_schematic.pdf + .png')
    plt.close(fig)


if __name__ == '__main__':
    main()
