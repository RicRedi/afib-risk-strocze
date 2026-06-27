# -*- coding: utf-8 -*-
"""
M1.2 — Forest plot binárních asociací s FiS
Výstupy: figures/forest_plot_binary.png/.pdf, tables/binary_associations.csv
Zdroj:   results/analysis_results_binary_variables_zachyt_FiS_mapping_correlations.json
         results/risk_scorecard.json  (pro identifikaci vybraných features)
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
TABLES  = Path(__file__).resolve().parent / 'tables'
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

BINARY_JSON = RESULTS / 'analysis_results_binary_variables_zachyt_FiS_mapping_correlations.json'
SCORECARD   = RESULTS / 'risk_scorecard.json'

with open(BINARY_JSON, encoding='utf-8') as f:
    data = json.load(f)
with open(SCORECARD, encoding='utf-8') as f:
    sc = json.load(f)

selected_features = set(sc['variables'].keys())

rows = []
for var, info in data.items():
    rows.append({
        'Proměnná':         var,
        'OR':               info['odds_ratio'],
        'CI_lower':         info['odds_CI_lower'],
        'CI_upper':         info['odds_CI_upper'],
        'p_hodnota':        info['p_value'],
        'Data použita (%)': info.get('data used [%]', 100.0),
        'Vybrána modelem':  var in selected_features,
    })

df = pd.DataFrame(rows).sort_values('OR', ascending=True).reset_index(drop=True)

csv_path = TABLES / 'binary_associations.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {csv_path}")

# --- Forest plot ---
n = len(df)
fig, ax = plt.subplots(figsize=(13, max(5, n * 0.65)))

# Crop long variable names for y-axis labels
def short_label(name, sel):
    s = name.replace('Osobní anamnéza (choice=', '').replace('Typ akutní ischemie (choice=', 'Ischemie: ') \
            .replace('Staré postischemické změny (choice=', 'Post-isch.: ') \
            .rstrip(')')
    return ('★ ' if sel else '   ') + s

labels = [short_label(row['Proměnná'], row['Vybrána modelem']) for _, row in df.iterrows()]

for i, row in df.iterrows():
    sig   = row['p_hodnota'] < 0.05
    sel   = row['Vybrána modelem']
    color = '#c0392b' if sig else '#95a5a6'
    lw    = 2.0 if sel else 1.2
    ms    = 9  if sel else 7

    err_low  = row['OR'] - row['CI_lower']
    err_high = row['CI_upper'] - row['OR']

    ax.errorbar(
        row['OR'], i,
        xerr=[[err_low], [err_high]],
        fmt='o', color=color, ecolor=color,
        capsize=4, capthick=lw, elinewidth=lw,
        markersize=ms,
        markeredgewidth=1.5 if sel else 0,
        markeredgecolor='black' if sel else color,
        zorder=3,
    )

    p_str = f"p={row['p_hodnota']:.3f}" if row['p_hodnota'] >= 0.001 else "p<0.001"
    annotation = f"OR {row['OR']:.2f} [{row['CI_lower']:.2f}–{row['CI_upper']:.2f}]  {p_str}"
    blend = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, i, annotation,
            transform=blend,
            fontsize=7.5, va='center', ha='left', clip_on=False)

ax.axvline(x=1.0, color='#2c3e50', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8.5)
for tick, is_sel in zip(ax.get_yticklabels(), df['Vybrána modelem']):
    tick.set_fontweight('bold' if is_sel else 'normal')

ax.set_xlabel('Odds Ratio (95% CI, Fisherův přesný test)', fontsize=10)
ax.set_title('Asociace binárních proměnných se záchytem FiS\n'
             '(červená = p < 0.05; ★ = zahrnuto v modelu)',
             fontsize=11, fontweight='bold', pad=10)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.set_xlim(left=0.05)
ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=1)
ax.text(0.0, -0.04,
        'Červená = p < 0.05  |  Šedá = p ≥ 0.05  |  ★ = vybrána modelem (RFECV)',
        transform=ax.transAxes, fontsize=8, color='#555555', style='italic')

plt.subplots_adjust(left=0.35, right=0.68, top=0.92, bottom=0.08)

for ext in ('png', 'pdf'):
    out = FIGURES / f'forest_plot_binary.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
