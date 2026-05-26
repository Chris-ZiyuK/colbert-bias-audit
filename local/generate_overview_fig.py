"""
TCD Pipeline Overview Figure — Clean version without unicode issues.
Uses DejaVu Sans which has full glyph coverage.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0,
    'mathtext.fontset': 'dejavusans',
})

fig = plt.figure(figsize=(14, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.0)
ax.axis('off')

# Colors
BG_BLUE = '#E3F2FD'
BG_GREEN = '#E8F5E9'
BG_ORANGE = '#FFF3E0'
BLUE = '#1565C0'
GREEN = '#2E7D32'
ORANGE = '#E65100'
RED = '#C62828'
FUNC = '#E64A19'   # Deep orange for function
CONT = '#1976D2'   # Blue for content
DARK = '#263238'
GRAY = '#78909C'
WHITE = '#FFFFFF'


def rbox(x, y, w, h, fc, ec=None, alpha=0.3, lw=1.5, zorder=1):
    if ec is None: ec = fc
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, **kw):
    d = dict(ha='center', va='center', fontsize=9, color=DARK, zorder=10)
    d.update(kw)
    return ax.text(x, y, s, **d)

def arrow(x1, y1, x2, y2, c=DARK, lw=2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=8)

# ══════════════════════════════════════════════════════════════
# PANEL 1: Counterfactual Setup
# ══════════════════════════════════════════════════════════════
rbox(0.1, 0.15, 4.2, 4.7, BG_BLUE, ec=BLUE, alpha=0.3, lw=2)
txt(2.2, 4.6, 'Step 1: Counterfactual Setup', fontsize=11, fontweight='bold', color=BLUE)

# Query
rbox(0.35, 3.7, 3.7, 0.65, WHITE, ec=BLUE, alpha=0.95, lw=1.5)
txt(2.2, 4.15, 'Query  Q', fontsize=8, fontweight='bold', color=BLUE)
txt(2.2, 3.85, '"Who is an experienced doctor?"', fontsize=8, fontstyle='italic')

# Doc A
rbox(0.35, 2.4, 3.7, 1.0, '#C8E6C9', ec=GREEN, alpha=0.5, lw=1.5)
txt(2.2, 3.2, 'Document  $D_A$', fontsize=8, fontweight='bold', color=GREEN)
# Name highlighted
rbox(0.55, 2.65, 0.7, 0.35, '#A5D6A7', ec=GREEN, alpha=0.8, lw=1.5)
txt(0.9, 2.82, 'Emily', fontsize=9, fontweight='bold', color=GREEN)
txt(2.55, 2.82, 'has over ten years of experience', fontsize=7.5, ha='left', color=DARK)
txt(2.2, 2.52, 'in this field and has worked with top organizations.', fontsize=7, color=GRAY)

# Doc B
rbox(0.35, 0.95, 3.7, 1.0, '#FFCDD2', ec=RED, alpha=0.5, lw=1.5)
txt(2.2, 1.75, 'Document  $D_B$', fontsize=8, fontweight='bold', color=RED)
rbox(0.55, 1.2, 0.85, 0.35, '#EF9A9A', ec=RED, alpha=0.8, lw=1.5)
txt(0.97, 1.37, 'Lakisha', fontsize=9, fontweight='bold', color=RED)
txt(2.65, 1.37, 'has over ten years of experience', fontsize=7.5, ha='left', color=DARK)
txt(2.2, 1.07, 'in this field and has worked with top organizations.', fontsize=7, color=GRAY)

# Swap indicator
rbox(0.35, 0.25, 3.7, 0.5, '#FFF9C4', ec='#F9A825', alpha=0.9, lw=1)
txt(2.2, 0.5, 'Only the name changes; all other text is identical', fontsize=7.5, fontweight='bold', color='#E65100')

# Double arrow between docs
ax.annotate('', xy=(0.2, 1.95), xytext=(0.2, 2.4),
            arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2.5), zorder=8)

# ══════════════════════════════════════════════════════════════
# PANEL 2: ColBERT MaxSim
# ══════════════════════════════════════════════════════════════
rbox(4.55, 0.15, 4.85, 4.7, BG_GREEN, ec=GREEN, alpha=0.2, lw=2)
txt(6.95, 4.6, 'Step 2: ColBERT MaxSim Scoring', fontsize=11, fontweight='bold', color=GREEN)

# Query tokens row
q_toks = ['Who', 'is', 'an', 'experi-\nenced', 'doctor', '?']
q_types = ['func','func','func','cont','cont','func']
q_labels = ['Who', 'is', 'an', 'expe-\nrienced', 'doctor', '?']
qx0 = 4.85
qsp = 0.75

txt(6.95, 4.25, 'Query tokens:', fontsize=8, fontweight='bold', color=DARK)

for i, (tok, typ) in enumerate(zip(q_labels, q_types)):
    x = qx0 + i*qsp
    c = FUNC if typ == 'func' else CONT
    bg = '#FFCCBC' if typ == 'func' else '#BBDEFB'
    rbox(x-0.28, 3.7, 0.56, 0.42, bg, ec=c, alpha=0.85, lw=1.5)
    txt(x, 3.91, tok, fontsize=7, fontweight='bold', color=c)

# Legend
rbox(4.85, 3.3, 1.6, 0.28, '#FFCCBC', ec=FUNC, alpha=0.7, lw=1)
txt(5.65, 3.44, 'Function word', fontsize=7, color=FUNC, fontweight='bold')
rbox(6.65, 3.3, 1.6, 0.28, '#BBDEFB', ec=CONT, alpha=0.7, lw=1)
txt(7.45, 3.44, 'Content word', fontsize=7, color=CONT, fontweight='bold')

# Per-token scores D_A
scores_a = [0.72, 0.68, 0.71, 0.85, 0.92, 0.65]
scores_b = [0.70, 0.64, 0.67, 0.84, 0.91, 0.63]
ya = 2.65
yb = 1.85

txt(5.3, ya+0.45, '$D_A$ scores (Emily):', fontsize=7.5, fontweight='bold', color=GREEN, ha='left')
for i, (sa, typ) in enumerate(zip(scores_a, q_types)):
    x = qx0 + i*qsp
    c = FUNC if typ=='func' else CONT
    bg = '#FFCCBC' if typ=='func' else '#BBDEFB'
    rbox(x-0.25, ya, 0.50, 0.35, bg, ec=c, alpha=0.7, lw=1)
    txt(x, ya+0.17, f'{sa:.2f}', fontsize=7, color=c, fontweight='bold')

txt(5.3, yb+0.45, '$D_B$ scores (Lakisha):', fontsize=7.5, fontweight='bold', color=RED, ha='left')
for i, (sb, typ) in enumerate(zip(scores_b, q_types)):
    x = qx0 + i*qsp
    c = FUNC if typ=='func' else CONT
    bg = '#FFCCBC' if typ=='func' else '#BBDEFB'
    rbox(x-0.25, yb, 0.50, 0.35, bg, ec=c, alpha=0.4, lw=1)
    txt(x, yb+0.17, f'{sb:.2f}', fontsize=7, color=c, fontweight='bold')

# MaxSim equation
rbox(4.85, 0.55, 4.3, 0.95, WHITE, ec=GREEN, alpha=0.95, lw=1.5)
txt(7.0, 1.15, 'MaxSim Relevance Score',
    fontsize=11, color=GREEN, fontweight='bold')
txt(7.0, 0.75, 'Relevance is the sum of maximum cosine similarity\nfor each query token across all document tokens.', fontsize=7, color=GRAY)

# ══════════════════════════════════════════════════════════════
# PANEL 3: TCD Attribution
# ══════════════════════════════════════════════════════════════
rbox(9.65, 0.15, 4.2, 4.7, BG_ORANGE, ec=ORANGE, alpha=0.2, lw=2)
txt(11.75, 4.6, 'Step 3: TCD Attribution', fontsize=11, fontweight='bold', color=ORANGE)

# TCD formula
rbox(9.85, 3.8, 3.8, 0.6, WHITE, ec=ORANGE, alpha=0.95, lw=1.5)
txt(11.75, 4.15, 'Token Contribution Disparity', fontsize=10.5, color=ORANGE, fontweight='bold')
txt(11.75, 3.9, 'TCD measures the absolute difference in MaxSim\ncontributions: |MaxSim(A) - MaxSim(B)|', fontsize=6.8, color=GRAY)

# TCD bars
tcds = [abs(a-b) for a,b in zip(scores_a, scores_b)]
max_t = max(tcds)
bx0 = 10.15
bsp = 0.6
by = 2.8

txt(11.75, 3.5, 'Per-token TCD values:', fontsize=8, fontweight='bold', color=DARK)

tok_labels = ['Who', 'is', 'an', 'exp.', 'doc.', '?']
for i, (tv, tok, typ) in enumerate(zip(tcds, tok_labels, q_types)):
    x = bx0 + i*bsp
    c = FUNC if typ=='func' else CONT
    bg = '#FFCCBC' if typ=='func' else '#BBDEFB'
    h = tv/max_t * 1.1 if max_t > 0 else 0
    ax.bar(x, h, width=0.42, bottom=by-h, color=bg, edgecolor=c, lw=1.5, zorder=5)
    txt(x, by+0.12, tok, fontsize=7, color=c, fontweight='bold')
    txt(x, by-h-0.13, f'{tv:.2f}', fontsize=6.5, color=c, fontweight='bold')

# Result box
rbox(9.85, 0.35, 3.8, 1.2, WHITE, ec=ORANGE, alpha=0.95, lw=1.5)

func_mean = np.mean([t for t,tp in zip(tcds, q_types) if tp=='func'])
cont_mean = np.mean([t for t,tp in zip(tcds, q_types) if tp=='cont'])

txt(10.85, 1.3, 'Func-TCD', fontsize=9, fontweight='bold', color=FUNC)
txt(10.85, 1.0, f'{func_mean:.3f}', fontsize=11, fontweight='bold', color=FUNC)

txt(11.75, 1.15, '>', fontsize=18, fontweight='bold', color=DARK)

txt(12.65, 1.3, 'Cont-TCD', fontsize=9, fontweight='bold', color=CONT)
txt(12.65, 1.0, f'{cont_mean:.3f}', fontsize=11, fontweight='bold', color=CONT)

txt(11.75, 0.6, 'Ratio: 1.43x (controlled)  /  2.08x (naturalistic-template)',
    fontsize=8, fontweight='bold', color=ORANGE)

# Panel arrows
arrow(4.15, 2.5, 4.7, 2.5, c=DARK, lw=2.5)
arrow(9.2, 2.5, 9.8, 2.5, c=DARK, lw=2.5)

plt.savefig('/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/paper/figures/fig0_tcd_overview.pdf',
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/paper/figures/fig0_tcd_overview.png',
            bbox_inches='tight', dpi=200, facecolor='white')
print("Done!")
