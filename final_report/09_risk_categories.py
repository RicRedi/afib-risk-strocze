# -*- coding: utf-8 -*-
"""
M2.4 — Distribuce rizikových kategorií a NPV
Výstup: figures/risk_category_distribution.png/.pdf
         tables/risk_categories_summary.csv
Zdroj:  results/risk_model_calibrated.joblib + x_test.npy + y_test.npy
"""
import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
TABLES  = Path(__file__).resolve().parent / 'tables'
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

x_test = np.load(RESULTS / 'risk_model_x_test.npy')
y_test = np.load(RESULTS / 'risk_model_y_test.npy')
model  = joblib.load(RESULTS / 'risk_model_calibrated.joblib')

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

y_prob     = model.predict_proba(x_test)[:, 1]
prevalence = sc['test_performance']['cohort_prevalence']

# --- Rizikové kategorie ---
CATEGORIES = [
    {'name': 'Nízké',        'label': 'Nízké\n(<1× prev.)',       'lo': 0.0,         'hi': 1.0 * prevalence, 'color': '#2ecc71'},
    {'name': 'Střední',      'label': 'Střední\n(1–1.5× prev.)',   'lo': 1.0 * prevalence, 'hi': 1.5 * prevalence, 'color': '#f39c12'},
    {'name': 'Vysoké',       'label': 'Vysoké\n(1.5–2× prev.)',    'lo': 1.5 * prevalence, 'hi': 2.0 * prevalence, 'color': '#e67e22'},
    {'name': 'Velmi vysoké', 'label': 'Velmi vysoké\n(>2× prev.)', 'lo': 2.0 * prevalence, 'hi': 1.0,            'color': '#e74c3c'},
]

rows = []
for cat in CATEGORIES:
    mask = (y_prob >= cat['lo']) & (y_prob < cat['hi'])
    if cat == CATEGORIES[-1]:
        mask = y_prob >= cat['lo']
    n_total = mask.sum()
    n_neg   = ((y_test == 0) & mask).sum()
    n_pos   = ((y_test == 1) & mask).sum()
    npv     = n_neg / n_total if n_total > 0 else float('nan')
    ppv     = n_pos / n_total if n_total > 0 else float('nan')
    rows.append({
        'Kategorie':    cat['name'],
        'Práh (prevalence ×)': f"{cat['lo']/prevalence:.1f}×–{cat['hi']/prevalence:.1f}×" if cat != CATEGORIES[-1] else '>2.0×',
        'Pravd. rozsah (%)': f"{cat['lo']*100:.1f} – {cat['hi']*100:.1f}" if cat != CATEGORIES[-1] else f">{cat['lo']*100:.1f}",
        'n (test)':     int(n_total),
        '% testovací sady': round(n_total / len(y_test) * 100, 1),
        'AFib+ v kategorii': int(n_pos),
        'NPV (%)':      round(npv * 100, 1) if not np.isnan(npv) else None,
        'PPV (%)':      round(ppv * 100, 1) if not np.isnan(ppv) else None,
        '_mask':        mask,
        '_color':       cat['color'],
        '_label':       cat['label'],
    })

df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows])
csv_path = TABLES / 'risk_categories_summary.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {csv_path}")
print(df.to_string(index=False))

# --- Graf ---
n_cats  = [r['n (test)'] for r in rows]
pcts    = [r['% testovací sady'] for r in rows]
npvs    = [r['NPV (%)'] for r in rows]
labels  = [r['_label'] for r in rows]
colors  = [r['_color'] for r in rows]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(rows))
bars = ax1.bar(x, n_cats, color=colors, alpha=0.85, width=0.55, zorder=3)

# Popisky nad sloupci
for i, (bar, n, pct, n_pos) in enumerate(zip(bars, n_cats, pcts, [r['AFib+ v kategorii'] for r in rows])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'n = {n}\n({pct} %)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('Počet pacientů (testovací sada)', fontsize=11)
ax1.set_title(
    'Distribuce pacientů v rizikových kategoriích záchytu FiS\n'
    f'(testovací sada, n = {len(y_test)}, prevalence = {prevalence*100:.1f} %)',
    fontsize=11, fontweight='bold', pad=10,
)
ax1.set_ylim(0, max(n_cats) * 1.28)
ax1.grid(axis='y', alpha=0.3, linestyle=':', zorder=1)

# Druhá osa — NPV
ax2 = ax1.twinx()
ax2.plot(x, npvs, 'o--', color='#2c3e50', lw=2.0, ms=9, zorder=5,
         label='NPV v kategorii')
for i, (xi, npv) in enumerate(zip(x, npvs)):
    ax2.text(xi + 0.22, npv + 0.4, f'{npv:.1f} %',
             fontsize=8.5, color='#2c3e50', va='bottom')

ax2.axhline(95, color='#8e44ad', ls=':', lw=1.2, alpha=0.7, label='95 % NPV (klinický práh)')
ax2.set_ylabel('NPV — % pacientů bez FiS v kategorii', fontsize=10)
ax2.set_ylim(80, 102)
ax2.legend(fontsize=9, loc='lower left')

# Prahové hodnoty jako anotace
prev_str = f'Kohortová prevalence: {prevalence*100:.1f} %'
ax1.text(0.01, 0.97, prev_str, transform=ax1.transAxes,
         fontsize=8.5, va='top', color='#555555', style='italic')

plt.tight_layout()

for ext in ('png', 'pdf'):
    out = FIGURES / f'risk_category_distribution.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
