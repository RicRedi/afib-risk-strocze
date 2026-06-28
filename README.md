# AFib Risk Kalkulačka — STROCZE kohorta

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Nástroj pro stratifikaci rizika záchytu fibrilace síní (FiS) u pacientů po ischemické CMP nebo TIA.
Model byl vytrénován na datech STROCZE kohorty a vybírá pacienty vhodné pro prodloužený
MDT monitoring FiS. Výstupem je kalibrovaná pravděpodobnost záchytu a jedna ze čtyř rizikových kategorií.

---

## English Abstract

This project identifies clinical predictors of atrial fibrillation (AFib) detection in post-stroke/TIA patients
using the Czech STROCZE registry. A logistic regression model with RFECV feature selection and Platt probability
calibration was trained on 1,884 complete patient records. The final model uses three inputs — CHA₂DS₂-VASc score,
territorial infarct type, and hyperlipidemia history — achieving AUC 0.647 [95% CI 0.543–0.740] on a held-out
test set (n = 377). A static HTML calculator is deployed publicly via GitHub Pages.

---

## HTML kalkulačka

**[ricredi.github.io/afib-risk-strocze](https://ricredi.github.io/afib-risk-strocze/)**

Nevyžaduje instalaci — funguje v libovolném prohlížeči.

---

## Quickstart

```bash
git clone https://github.com/RicRedi/afib-risk-strocze.git
cd afib-risk-strocze
pip install -r requirements.txt

python predict.py          # Python demo — 3 ukázkové predikce
python main.py             # Explorační analýza (vyžaduje zdrojová data v src/)
```

---

## Struktura repozitáře

```
afib-risk-strocze/
├── main.py                    vstupní bod explorační analýzy
├── analyze.py                 VariableCorrelationAnalyzer
├── core.py                    příprava dat, IQR filtr, binarizace
├── risk_model.py              RiskScoreGenerator (RFECV, LR, Platt scaling)
├── predict.py                 Python demo — inference pro 3 pacienty
├── plotting.py                vizualizační engine
│
├── config/                    YAML konfigurace (variables, model, analysis)
├── utils/                     ConfigSingleton a helper funkce
│
├── final_report/              vizualizační skripty (01–12) + figures/ + tables/
├── docs/tacr/                 technická dokumentace (5 dokumentů, čeština)
│
├── web/                       HTML kalkulačka (GitHub Pages)
│   ├── index.html
│   ├── assets/scorecard.js    AUTO-GENERATED — spustit web/generate_scorecard_js.py
│   └── assets/style.css
│
└── results/
    ├── risk_scorecard.json    koeficienty a metriky modelu
    ├── risk_model_x_test.npy  testovací data (377 pacientů, 3 features, bez PII)
    └── risk_model_y_test.npy  testovací labely (0/1)
```

---

## Model — klíčové metriky

| Metrika | Hodnota | 95% CI |
|---------|---------|--------|
| AUC-ROC | 0.647 | [0.543–0.740] |
| Senzitivita (Youden) | 95.5 % | — |
| Specificita (Youden) | 30.1 % | — |
| NPV (kategorie Nízké) | 95.7 % | — |
| Brier (kalibrovaný) | 0.055 | — |
| EPV | 29.7 | — |

Test set: n = 377, AFib+ = 22 (5.8 %). Kalibrace: Platt scaling (sigmoid).

---

## Dokumentace

Technická dokumentace v češtině je dostupná ve složce [docs/tacr/](docs/tacr/):

| Dokument | Obsah |
|----------|-------|
| [uzivatelska_prirucka.md](docs/tacr/uzivatelska_prirucka.md) | Jak používat kalkulačku, interpretace výsledků |
| [technicka_dokumentace.md](docs/tacr/technicka_dokumentace.md) | Architektura systému, algoritmy |
| [programatorska_dokumentace.md](docs/tacr/programatorska_dokumentace.md) | API, konfigurace, rozšíření |
| [analyza_funkcionalnich_pozadavku.md](docs/tacr/analyza_funkcionalnich_pozadavku.md) | Funkční a nefunkční požadavky |
| [overeni_funkcnosti.md](docs/tacr/overeni_funkcnosti.md) | Testovací strategie, výsledky 101 testů |

---

## Citace

Projekt vznikl v rámci výzkumu na [Vysokém učení technickém v Brně](https://www.vut.cz/) a [Medical Data Transfer s.r.o.](https://www.mdt.cz/).
Data pocházejí z kohorty STROCZE (STROke Czech REgistry).

**Autoři:**  
- **Valentýna Provazník**  
E-mail: provaznik@vut.cz  
Vysoké učení technické v Brně  

- **Veronika Bulková**  
E-mail: veronika.bulkova@gmail.com  
Medical Data Transfer s.r.o.

- **Richard Ředina**  
E-mail: 195715@vut.cz  
Vysoké učení technické v Brně

---

## Upozornění

Tento nástroj byl vytvořen výhradně pro výzkumné účely v rámci projektu NCK FEIM - DP001N (Healthcare).
**Není schválen pro klinické rozhodování.** Klinická odpovědnost zůstává na ošetřujícím lékaři.
Před nasazením v klinické praxi je nutná prospektivní validace na nezávislé kohortě.
