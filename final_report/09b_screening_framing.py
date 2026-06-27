# -*- coding: utf-8 -*-
"""
M2.4b — Screeningová efektivita modelu (NNM + rule-out framing)
Výstup: figures/screening_efficiency.png/.pdf
Zdroj:  results/risk_scorecard.json
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
FIGURES.mkdir(exist_ok=True)

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

cal    = sc['calibration']
tp, tn = cal['calibrated_tp'], cal['calibrated_tn']
fp, fn = cal['calibrated_fp'], cal['calibrated_fn']
thr    = cal['calibrated_optimal_threshold']
npv    = cal['calibrated_npv']
sens   = cal['calibrated_sensitivity']
n_test = sc['test_performance']['n_test']
n_pos  = sc['test_performance']['n_positive']
n_neg  = sc['test_performance']['n_negative']

pred_pos = tp + fp          # 269 — monitorováni
pred_neg = tn + fn          # 108 — vyloučeni
nnm_base  = n_test / n_pos
nnm_model = pred_pos / tp
nnm_delta = (nnm_base - nnm_model) / nnm_base

print(f"Bez modelu:  monitor {n_test}, záchyt {n_pos}/{n_pos}, NNM = {nnm_base:.1f}")
print(f"S modelem:   monitor {pred_pos}, záchyt {tp}/{n_pos}, NNM = {nnm_model:.1f}")
print(f"Vyloučeno:   {pred_neg} pac., FN = {fn}, NPV = {npv*100:.1f} %")
print(f"Úspora:      {pred_neg/n_test*100:.1f} %  (NNM −{nnm_delta:.0%})")

# =============================================================================
# LAYOUT: suptitle / [bar  |  donut] / [shared legend] / [footer text]
# Vyřešeno statickým subplots_adjust — žádné tight_layout
# =============================================================================
C_TP = '#1a5276'   # tmavě modrá  — AFib zachycena (TP)
C_FP = '#aed6f1'   # světle modrá — zbytečný monitoring (FP)
C_TN = '#d5f5e3'   # světle zelená — správně vyloučena (TN)
C_FN = '#c0392b'   # červená      — přehlédnutá FiS (FN)

fig = plt.figure(figsize=(14, 7))

gs = GridSpec(1, 2, figure=fig,
              width_ratios=[1.55, 1],
              left=0.23, right=0.97,
              top=0.86, bottom=0.28,
              wspace=0.14)

ax_bar  = fig.add_subplot(gs[0])
ax_rule = fig.add_subplot(gs[1])

# ── PANEL A: stacked horizontal bars ─────────────────────────────────────────

y    = np.array([1.25, 0.0])
bar_h = 0.48

def hbar(ax, y_pos, xstart, width, color, **kw):
    ax.barh(y_pos, width, left=xstart, height=bar_h, color=color, **kw)
    return xstart + width

# Bez modelu
hbar(ax_bar, y[0], 0,     n_pos, C_TP, edgecolor='white', linewidth=0.5)
hbar(ax_bar, y[0], n_pos, n_neg, C_FP, edgecolor='white', linewidth=0.5)

ax_bar.text(n_pos + n_neg / 2, y[0],
            f'{n_neg} zbytečných monitoringů',
            ha='center', va='center', fontsize=9, color='#1a3a5c')

# S modelem
hbar(ax_bar, y[1], 0,           tp, C_TP, edgecolor='white', linewidth=0.5)
hbar(ax_bar, y[1], tp,          fp, C_FP, edgecolor='white', linewidth=0.5)
hbar(ax_bar, y[1], tp + fp,     tn, C_TN, edgecolor='#27ae60', linewidth=1.4, linestyle='--')
hbar(ax_bar, y[1], tp + fp + tn, fn, C_FN, edgecolor='white', linewidth=0.5)

ax_bar.text(tp + fp / 2, y[1],
            f'{fp} zbytečných monitoringů',
            ha='center', va='center', fontsize=9, color='#1a3a5c')
ax_bar.text(tp + fp + tn / 2, y[1],
            f'{tn} správně\nvyloučeno',
            ha='center', va='center', fontsize=8.5, color='#1e8449', fontweight='bold')

# NNM labely vpravo
x_nnm = n_test + 8
ax_bar.text(x_nnm, y[0],
            f'NNM = {nnm_base:.1f}\n({n_pos}/{n_pos} AFib)',
            va='center', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')
ax_bar.text(x_nnm, y[1],
            f'NNM = {nnm_model:.1f}\n({tp}/{n_pos} AFib)',
            va='center', ha='left', fontsize=10, fontweight='bold', color='#2980b9')

# Šipka mezi NNM labely
x_arr = x_nnm - 2
ax_bar.annotate('', xy=(x_arr, y[1] + bar_h / 2 + 0.06),
                xytext=(x_arr, y[0] - bar_h / 2 - 0.06),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
ax_bar.text(x_arr + 2, (y[0] + y[1]) / 2,
            f'NNM\n−{nnm_delta:.0%}',
            va='center', ha='left', fontsize=9, color='#27ae60', fontweight='bold')

# Osy
ax_bar.set_yticks(y)
ax_bar.set_yticklabels(
    [f'Bez modelu\n(monitor všichni, n = {n_test})',
     f'S modelem\n(práh = {thr:.3f})'],
    fontsize=10,
)
ax_bar.tick_params(axis='y', pad=8)
ax_bar.set_xlabel('Počet pacientů', fontsize=10)
ax_bar.set_xlim(0, n_test + 70)
ax_bar.set_ylim(-0.55, 1.80)
ax_bar.set_title('A — Monitoring burden: bez modelu vs. s modelem',
                 fontsize=10.5, fontweight='bold', loc='left', pad=7)
ax_bar.grid(axis='x', alpha=0.25, linestyle=':')
ax_bar.spines[['top', 'right']].set_visible(False)

# ── PANEL B: donut NPV ───────────────────────────────────────────────────────

wedges, _ = ax_rule.pie(
    [tn, fn],
    colors=[C_TN, C_FN],
    explode=[0.03, 0.12],
    startangle=90,
    wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2),
    counterclock=False,
)

ax_rule.text(0, 0.10, f'{npv*100:.1f} %',
             ha='center', va='center', fontsize=30, fontweight='bold', color='#1e8449')
ax_rule.text(0, -0.24, 'NPV',
             ha='center', va='center', fontsize=14, color='#555')

# Jednoduché textové popisky — bez legend, bez šipek
ax_rule.text(0, -1.50,
             f'● {tn} správně vyloučeno (TN, {tn/pred_neg*100:.1f} %)',
             ha='center', va='center', fontsize=9, color='#1e8449', fontweight='bold')
ax_rule.text(0, -1.80,
             f'● {fn} přehlédnutá FiS (FN, {fn/pred_neg*100:.1f} %)',
             ha='center', va='center', fontsize=9, color='#c0392b', fontstyle='italic')

ax_rule.set_xlim(-1.5, 1.5)
ax_rule.set_ylim(-2.05, 1.35)
ax_rule.set_title(f'B — Rule-out výkonnost\n({pred_neg} pac. s predikovanou negativitou)',
                  fontsize=10.5, fontweight='bold', pad=7)

# ── SDÍLENÁ LEGENDA (pod oběma panely) ──────────────────────────────────────
patches = [
    mpatches.Patch(color=C_TP, label=f'AFib+ zachycena — TP (n = {tp})'),
    mpatches.Patch(color=C_FP, label=f'AFib− → zbytečný monitoring — FP (n = {fp})'),
    mpatches.Patch(color=C_TN, label=f'AFib− správně vyloučena — TN (n = {tn})'),
    mpatches.Patch(color=C_FN, label=f'AFib+ přehlédnuta — FN (n = {fn})'),
]
fig.legend(handles=patches, fontsize=8.5, ncol=4,
           loc='lower center', bbox_to_anchor=(0.5, 0.12),
           framealpha=0.95, edgecolor='#cccccc')

# ── FOOTER (pod legendou) ────────────────────────────────────────────────────
fig.text(0.5, 0.08,
         f'Z {pred_neg} pacientů doporučených k vyloučení z MDT monitoringu je AFib+ jen {fn} '
         f'(NPV = {npv*100:.1f} %). Při senzitivitě {sens*100:.0f} % model ušetří '
         f'{pred_neg} monitoringů ({pred_neg/n_test*100:.0f} %) — NNM klesá o {nnm_delta:.0%}.',
         ha='center', va='bottom', fontsize=8.5, color='#444',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1',
                   edgecolor='#27ae60', alpha=0.88))

# ── HLAVNÍ TITULEK ───────────────────────────────────────────────────────────
fig.text(0.5, 0.94,
         'Kalibrovaný model jako screeningový (rule-out) nástroj',
         ha='center', va='bottom', fontsize=13, fontweight='bold', color='#1a252f')
fig.text(0.5, 0.90,
         f'NNM: {nnm_base:.1f} (bez modelu)  →  {nnm_model:.1f} (s modelem)  '
         f'|  senzitivita {sens*100:.0f} %  |  úspora {pred_neg/n_test*100:.0f} % monitoringů',
         ha='center', va='bottom', fontsize=9.5, color='#555')

# ── EXPORT ───────────────────────────────────────────────────────────────────
for ext in ('png', 'pdf'):
    out = FIGURES / f'screening_efficiency.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
