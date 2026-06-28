# Programátorská dokumentace
## Kalkulátor rizika záchytu fibrilace síní po iCMP/TIA

---

**Projekt:** NCK FEIM - DP001N (Healthcare) — Identifikace rizikových faktorů fibrilace síní  
**Verze dokumentu:** 1.0  
**Datum:** 2026-06-27  
**Autoři:** Valentýna Provazník, Veronika Bulková, Richard Ředina  
**Instituce:** Vysoké učení technické v Brně / Medical Data Transfer s.r.o.

---

## 1. Požadavky prostředí

| Požadavek | Verze | Poznámka |
|-----------|-------|----------|
| Python | ≥ 3.11 | Testováno na 3.12 |
| scikit-learn | ≥ 1.6 | RFECV, CalibratedClassifierCV |
| pandas | ≥ 2.0 | Zpracování dat |
| numpy | ≥ 1.26 | Numerické operace |
| joblib | ≥ 1.3 | Serializace modelů |
| matplotlib | ≥ 3.8 | Vizualizace (final_report/) |
| openpyxl | ≥ 3.1 | Čtení Excel souborů |
| pyyaml | ≥ 6.0 | Načítání konfigurace |
| statsmodels | ≥ 0.14 | Logistická regrese v analyze.py |
| scipy | ≥ 1.12 | Fisherův test, statistické testy |

Kompletní seznam závislostí je v souboru `requirements.txt` v kořeni repozitáře.

---

## 2. Instalace a spuštění

### 2.1 Klonování repozitáře

```bash
git clone https://github.com/RicRedi/afib-risk-strocze.git
cd afib-risk-strocze
```

### 2.2 Vytvoření virtuálního prostředí

```bash
# Windows (PowerShell)
python -m venv .env
.\.env\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .env
source .env/bin/activate
```

### 2.3 Instalace závislostí

```bash
pip install -r requirements.txt
```

### 2.4 Ověření instalace

```bash
python -c "from risk_model import RiskScoreGenerator; print('OK')"
```

### 2.5 Spuštění demo predikce

```bash
python predict.py
```

Při prvním spuštění bez uloženého modelu (`results/risk_model_calibrated.joblib`) skript automaticky spustí trénink. Trénink trvá přibližně 30–60 sekund v závislosti na hardware.

---

## 3. Struktura repozitáře

```
afib-risk-strocze/
├── README.md                          # Vstupní bod projektu
├── requirements.txt                   # Python závislosti
├── .gitignore                         # Vylučuje src/, *.joblib, .env/
│
├── config/                            # Konfigurační YAML soubory
│   ├── config.yaml                    # Hlavní agregátor konfigurace
│   ├── analysis.yaml                  # Parametry explorační analýzy
│   ├── variables.yaml                 # Definice proměnných
│   ├── model.yaml                     # Parametry modelu a cesty výstupů
│   └── plotting.yaml                  # Nastavení vizualizace
│
├── core.py                            # Utility: načítání dat, IQR filtrace
├── analyze.py                         # VariableCorrelationAnalyzer
├── risk_model.py                      # RiskScoreGenerator (hlavní ML pipeline)
├── predict.py                         # Demo inference — 3 ukázkové pacienty
├── plotting.py                        # CorrelationPlotter
├── utils/
│   ├── config_singleton.py            # Singleton pro konfiguraci
│   └── load_config.py                 # YAML loader
│
├── final_report/                      # Vizualizační skripty (M1–M4)
│   ├── 01_cohort_description.py       # Popis kohorty
│   ├── 02_exploratory_binary.py       # Forest plot binárních asociací
│   ├── 03_exploratory_continuous.py   # Kontinuální asociace
│   ├── 04_significant_boxplots.py     # Boxploty
│   ├── 05_rfecv_curve.py              # RFECV křivka
│   ├── 06_model_coefficients.py       # Koeficienty modelu
│   ├── 07_roc_curve.py                # ROC křivka s CI
│   ├── 08_calibration_plot.py         # Kalibrační diagram
│   ├── 09_risk_categories.py          # Rizikové kategorie
│   ├── 09b_screening_framing.py       # Screeningová efektivita (NNM)
│   ├── 10_model_comparison.py         # Srovnání 1/2/3 featurových modelů
│   ├── 11_confusion_matrix.py         # Matice záměn
│   ├── 12_threshold_sensitivity.py    # Citlivostní analýza prahů
│   ├── figures/                       # Výstupní PNG + PDF grafy
│   └── tables/                        # Výstupní CSV tabulky
│
├── docs/tacr/                         # Tato dokumentace (5 souborů)
│
├── web/                               # HTML kalkulačka (GitHub Pages)
│   ├── index.html                     # Vstupní stránka
│   ├── .nojekyll                      # Zakazuje Jekyll zpracování
│   ├── generate_scorecard_js.py       # Generátor scorecard.js ze scorecardu
│   └── assets/
│       ├── scorecard.js               # AUTO-GENERATED — needitovat ručně
│       └── style.css                  # Styly kalkulačky
│
└── results/                           # Výstupy modelu
    ├── risk_scorecard.json            # ✓ V Gitu: koeficienty a metriky
    ├── risk_model_x_test.npy          # ✓ V Gitu: testovací features (3 čísla/pac.)
    ├── risk_model_y_test.npy          # ✓ V Gitu: testovací labely (0/1)
    ├── risk_model.joblib              # ✗ Gitignored: základní LR model
    └── risk_model_calibrated.joblib   # ✗ Gitignored: kalibrovaný model
```

---

## 4. Popis hlavních tříd a metod

### 4.1 `RiskScoreGenerator` (`risk_model.py`)

Hlavní ML pipeline. Zapouzdřuje celý životní cyklus modelu od načtení dat po export.

#### Klíčové metody

| Metoda | Popis | Vstup | Výstup |
|--------|-------|-------|--------|
| `train()` | Kompletní trénink pipeline — volá všechny privátní metody v pořadí | — | Nastaví atributy instance |
| `predict_risk(patient_dict)` | Inference pro jednoho pacienta | `dict` s hodnotami prediktorů | `dict` s pravděpodobností a kategorií |
| `export_scorecard()` | Uloží modely (.joblib) a scorecard (.json) | — | Soubory v `results/` |
| `_load_and_prepare_data()` | Načte Excel, aplikuje IQR filtraci a binárization | — | Nastaví `self.df` |
| `_split_data()` | Stratifikovaný 80/20 split | — | Nastaví `self.x_train`, `self.x_test` atd. |
| `_select_features_rfecv()` | RFECV výběr prediktorů | — | Nastaví `self.selected_features` |
| `_train_model()` | Trénink LogisticRegression | — | Nastaví `self.model` |
| `_run_cross_validation()` | 5-fold CV na trénovací sadě | — | Nastaví `self.cv_results` |
| `_calibrate_model()` | Platt scaling na testovacích datech | — | Nastaví `self.calibrated_model` |
| `_compute_bootstrap_ci()` | 1 000× stratifikovaný bootstrap CI pro AUC, senzitivitu, specificitu | — | Nastaví CI hodnoty v `self.test_performance` |
| `_compute_risk_categories()` | Rozdělí testovací sadu do 4 kategorií | — | Nastaví kategorie v scorecardu |

#### Kompletní příklad end-to-end tréninku

```python
from utils.config_singleton import ConfigSingleton
from risk_model import RiskScoreGenerator

ConfigSingleton.set()        # Načtení konfigurace z config/*.yaml
gen = RiskScoreGenerator()
gen.train()                  # ~30–60 sekund
gen.export_scorecard()       # Uloží výsledky do results/
```

### 4.2 `VariableCorrelationAnalyzer` (`analyze.py`)

Explorační analýza — testuje statistické asociace mezi klinickými proměnnými a záchytem FiS.

| Metoda | Popis |
|--------|-------|
| `pipeline()` | Spustí kompletní analýzu (kontinuální + binární proměnné) |
| `analyze_continuous()` | Logistická regrese pro 19 kontinuálních proměnných |
| `analyze_binary()` | Chi-square / Fisherův test pro 12 binárních proměnných |

### 4.3 Inference funkce (`predict.py`)

Tyto funkce jsou určeny pro přímé volání při produkčním nasazení nebo v demo skriptu.

| Funkce | Popis | Vstup | Výstup |
|--------|-------|-------|--------|
| `predict_with_model(mdl, patient_dict, features)` | Inference ze sklearn modelu | Fitted estimator, dict, list | `float` — pravděpodobnost |
| `predict_from_scorecard(scorecard, patient_dict)` | Bodový výpočet ze scorecard.json | dict scorecard, dict pacienta | dict s body a breakdown |
| `risk_category(probability, prevalence)` | Mapování pravděpodobnosti na kategorii | float, float | dict s kategorií a popisem |
| `load_model_and_scorecard(mdl_path, sc_path)` | Načtení modelu a scorecardu z disku | str, str | tuple (model, dict) |

---

## 5. Konfigurační soubory

Projekt používá YAML konfiguraci načítanou přes `ConfigSingleton` (singleton pattern — konfigurace je načtena jednou a dostupná globálně).

### 5.1 `config/analysis.yaml`

```yaml
file_path: ./src/STROCZECHMDTHolterEK_DATA_LABELS_2025-05-21_1136p.xls
iqr_threshold: 1.5              # Násobek IQR pro filtraci outlierů
significance_level: 0.05        # Hladina významnosti p-hodnot
save_results: True
save_path: ./results
```

### 5.2 `config/variables.yaml`

Definuje seznam analyzovaných proměnných, cílovou proměnnou a podmínky filtrování. Klíče `independent_continuous_variables` a `independent_binary_variables` určují, které proměnné jsou analyzovány v `VariableCorrelationAnalyzer`.

### 5.3 `config/model.yaml`

Klíčové parametry modelu:

```yaml
model_features:                        # Kandidátní prediktory pro RFECV
  - "CHA₂DS₂-VASc"
  - "Typ akutní ischemie (choice=Teritoriální)"
  - "Osobní anamnéza (choice=Hyperlipidémie)"
  - "Enddiastolický rozměr levé komory (LVEDD)"
  - "Interventrikulární septum (IVS)"
  - "Ejekční frakce levé komory (LVEF)"
  - "BMI"
  - "Sérový kreatinin"

reference_var: "Záchyt FiS MDT celkově"

model_output:
  model_path: ./results/risk_model.joblib
  scorecard_path: ./results/risk_scorecard.json

training:
  test_size: 0.2
  random_state: 42
  scaling_factor: 10             # Koeficienty jsou škálovány × 10 pro bodové skóre
  n_bootstrap: 1000
  class_weight: balanced
```

---

## 6. Generování HTML kalkulačky

Po každém re-tréninku modelu (pokud se změní koeficienty) je nutné aktualizovat JavaScript soubor kalkulačky:

```bash
python web/generate_scorecard_js.py
```

Skript načte `results/risk_scorecard.json` a vygeneruje `web/assets/scorecard.js` s aktuálními koeficienty a kalibračními parametry:

```javascript
// AUTO-GENERATED — needitovat ručně
const SCORECARD = {
  lr_intercept: -0.7829,
  lr_coefs: {
    "CHA2DS2VASc": 1.1014,
    "Teritorialni": 1.0317,
    "Hyperlipidemie": -1.0
  },
  platt_a: -0.8019,
  platt_b: 2.7756,
  prevalence: 0.05836
};
```

Kalkulační vzorec v JavaScriptu:
```javascript
function predict(cha2ds2, terit, hlp) {
  const logit = SCORECARD.lr_intercept
    + SCORECARD.lr_coefs.CHA2DS2VASc * cha2ds2
    + SCORECARD.lr_coefs.Teritorialni * terit
    + SCORECARD.lr_coefs.Hyperlipidemie * hlp;
  const prob = 1 / (1 + Math.exp(-(SCORECARD.platt_a * logit + SCORECARD.platt_b)));
  const ratio = prob / SCORECARD.prevalence;
  return { prob, ratio, category: getCategory(ratio) };
}
```

---

## 7. Jak přidat nový prediktor

1. Ověřte, že nová proměnná existuje v Excel souboru se správným názvem sloupce.
2. Přidejte název do `config/model.yaml → model_features`.
3. Pokud je proměnná binární (ne/ano → 0/1), ověřte, že `core.py → convert_column_to_binary()` ji správně převede.
4. Spusťte re-trénink: `python risk_model.py` — RFECV automaticky rozhodne, zda je nová proměnná dostatečně informativní pro zařazení do modelu.
5. Po re-tréninku spusťte `python web/generate_scorecard_js.py` pro aktualizaci HTML kalkulačky.

---

## 8. Vizualizační skripty (final_report/)

Každý skript je samostatný a spouští se z kořene repozitáře:

```bash
python final_report/07_roc_curve.py
```

Skripty načítají data výhradně ze souboru `results/risk_scorecard.json` a z NumPy polí `x_test.npy`, `y_test.npy` — **netrénují model znovu**. Výstupy jsou ukládány do `final_report/figures/` (PNG + PDF, 300 DPI) a `final_report/tables/` (CSV).

---

*Dokument připraven v rámci projektu NCK FEIM - DP001N (Healthcare). Verze 1.0, 2026-06-27.*
