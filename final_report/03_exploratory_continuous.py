# -*- coding: utf-8 -*-
"""
M1.3 — Tabulka kontinuálních asociací s FiS
Výstup: tables/continuous_associations.csv
Zdroj:  results/analysis_results_continuous_variables_zachyt_FiS_mapping_correlations.json
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
TABLES  = Path(__file__).resolve().parent / 'tables'
TABLES.mkdir(exist_ok=True)

CONT_JSON = RESULTS / 'analysis_results_continuous_variables_zachyt_FiS_mapping_correlations.json'

with open(CONT_JSON, encoding='utf-8') as f:
    data = json.load(f)

rows = []
for var, info in data.items():
    p = info['p_value']
    rows.append({
        'Proměnná':       var,
        'β (log-OR)':     round(info['correlation'], 5),
        'p-hodnota':      round(p, 6),
        'p < 0.05':       'ANO' if p < 0.05 else 'ne',
        'Typ testu':      info.get('type', 'logistic'),
        'Data použita (%)': info.get('data used [%]', '?'),
    })

df = pd.DataFrame(rows).sort_values('p-hodnota', ascending=True).reset_index(drop=True)

csv_path = TABLES / 'continuous_associations.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {csv_path}")
print(f"\nSignifikantní proměnné (p < 0.05): {(df['p < 0.05'] == 'ANO').sum()} z {len(df)}")
print(df.to_string(index=False))
