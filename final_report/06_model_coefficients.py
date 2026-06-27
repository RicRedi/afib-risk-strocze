# -*- coding: utf-8 -*-
"""
M2.1 — Koeficienty logistického regresního modelu
Výstup: figures/model_coefficients.png/.pdf
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
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
TABLES  = Path(__file__).resolve().parent / 'tables'
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

SCALING = sc['scaling_factor']   # 10

# --- Příprrava dat ---
SHORT_LABELS = {
    'CHA₂DS₂-VASc':                                    'CHA₂DS₂-VASc',
    'Typ akutní ischemie (choice=Teritoriální)':        'Teritoriální\ninfarkt',
    'Osobní anamnéza (choice=Hyperlipidémie)':          'Hyperlipidémie\n(osobní anamnéza)',
}

vars_data = []
for var, info in sc['variables'].items():
    log_or = info['points_per_unit'] / SCALING
    vars_data.append({
        'label':   SHORT_LABELS.get(var, var),
        'full':    var,
        'log_or':  log_or,
        'points':  info['points_per_unit'],
        'paradox': 'Hyperlipidémie' in var,
    })

# Seřazení: sestupně podle log-OR (největší nahoře)
vars_data.sort(key=lambda x: x['log_or'], reverse=True)

# --- Tabulka ---
TYPE_MAP = {
    'CHA₂DS₂-VASc': 'kontinuální',
    'Typ akutní ischemie (choice=Teritoriální)': 'binární',
    'Osobní anamnéza (choice=Hyperlipidémie)': 'binární',
}
tbl_rows = []
for d in vars_data:
    lo = d['log_or']
    tbl_rows.append({
        'Feature':          d['full'],
        'β (log-OR)':       round(lo, 5),
        'OR (e^β)':         round(np.exp(lo), 4),
        'Body/jednotku':    round(d['points'], 2),
        'Typ':              TYPE_MAP.get(d['full'], '?'),
        'Poznámka':         'treatment paradox*' if d['paradox'] else '',
    })
tbl = pd.DataFrame(tbl_rows)
tbl_path = TABLES / 'model_scorecard.csv'
tbl.to_csv(tbl_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {tbl_path}")
print(tbl.to_string(index=False))

labels  = [d['label']  for d in vars_data]
log_ors = [d['log_or'] for d in vars_data]
points  = [d['points'] for d in vars_data]
colors  = ['#2980b9' if v > 0 else '#e74c3c' for v in log_ors]

n = len(vars_data)

# --- Graf ---
fig, ax = plt.subplots(figsize=(9, 4))

bars = ax.barh(range(n), log_ors, color=colors, alpha=0.82, height=0.55, zorder=3)

# Hodnoty na konci každé tyčky
for i, (bar, lo, pts) in enumerate(zip(bars, log_ors, points)):
    x_end = bar.get_width()
    offset = 0.04 if x_end >= 0 else -0.04
    ha     = 'left' if x_end >= 0 else 'right'
    sign   = '+' if pts > 0 else ''
    ax.text(
        x_end + offset, i,
        f'log-OR {lo:+.3f}  ({sign}{pts:.1f} bodů)',
        va='center', ha=ha, fontsize=8.5,
        color='#2c3e50',
    )

# Treatment paradox šipka + poznámka
paradox_idx = next(i for i, d in enumerate(vars_data) if d['paradox'])
ax.annotate(
    'Léčebný paradox:\nstatiny (nepotvrzeno v datech)',
    xy=(log_ors[paradox_idx], paradox_idx),
    xytext=(log_ors[paradox_idx] - 0.35, paradox_idx - 0.55),
    fontsize=7.5,
    color='#c0392b',
    style='italic',
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
    ha='center',
    va='top',
)

ax.axvline(0, color='#2c3e50', linewidth=1.0, linestyle='--', alpha=0.6, zorder=2)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Log-Odds koeficient (logistická regrese)', fontsize=10)
ax.set_title(
    'Koeficienty modelu — vybrané prediktory záchytu FiS\n'
    '(RFECV z 8 kandidátů; třída vyvážena class_weight="balanced")',
    fontsize=11, fontweight='bold', pad=10,
)
ax.set_xlim(min(log_ors) - 0.7, max(log_ors) + 1.2)
ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=1)

# Legenda
pos_patch = mpatches.Patch(color='#2980b9', alpha=0.82, label='Pozitivní prediktor (↑ riziko FiS)')
neg_patch = mpatches.Patch(color='#e74c3c', alpha=0.82, label='Negativní prediktor (↓ riziko FiS)')
ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc='lower left')

# Intercept poznámka
ax.text(
    0.01, -0.12,
    f'Intercept: {sc["intercept"]:.4f}  |  Škálovací faktor: {SCALING}  |  '
    f'Skóre = koeficient × {SCALING}',
    transform=ax.transAxes, fontsize=7.5, color='#777777', style='italic',
)

plt.tight_layout(rect=[0, 0.06, 1, 1])

for ext in ('png', 'pdf'):
    out = FIGURES / f'model_coefficients.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
