# Analýza funkčních požadavků
## Kalkulátor rizika záchytu fibrilace síní po iCMP/TIA

---

**Projekt:** NCK FEIM - DP001N (Healthcare) — Identifikace rizikových faktorů fibrilace síní  
**Verze dokumentu:** 1.0  
**Datum:** 2026-06-27  
**Autoři:** Valentýna Provazník, Veronika Bulková, Richard Ředina  
**Instituce:** Vysoké učení technické v Brně / Medical Data Transfer s.r.o.

---

## 1. Kontext a motivace

### 1.1 Klinický problém

Fibrilace síní (FiS) je nejčastější porucha srdečního rytmu a příčina 20–30 % ischemických cévních mozkových příhod (iCMP). U pacientů po iCMP nebo tranzitorní ischemické atace (TIA) je záchyt dosud neznámé FiS klíčový pro volbu sekundárně preventivní terapie: při potvrzené FiS je indikována antikoagulační léčba, která snižuje riziko recidivy CMP o 60–70 %.

Diagnostika FiS je obtížná, protože FiS může být paroxyzmální (záchvatovitá) a standardní 12-svodové EKG ji zachytí pouze tehdy, probíhá-li právě v okamžiku záznamu. Dlouhodobý MDT (Medical Data Transfer) monitoring — 24–48hodinové Holter EKG záznamy prováděné firmou MDT Brno — výrazně zvyšuje senzitivitu záchytu.

### 1.2 Stávající praxe a její limitace

V běžné klinické praxi jsou k dlouhodobé holterovské monitoraci indikováni pacienti na základě klinické indikace a hodnoty CHA₂DS₂-VASc skóre. Skóre CHA₂DS₂-VASc ≥ 2 je obecně akceptovanou indikací k antikoagulační léčbě, avšak nerozlišuje, kteří konkrétní pacienti s touto hodnotou mají vyšší pravděpodobnost záchytu FiS při monitoraci. Tím dochází k:

- **Přetěžování kapacit holterovské monitorace:** Pacienti s nízkým skutečným rizikem FiS podstupují monitoraci zbytečně.
- **Nerovnoměrné alokaci zdrojů:** Pacienti s vysokým rizikem nemusí být prioritně vyšetřeni.
- **Suboptimálnímu poměru náklady/přínos:** Záchyt 1 případu FiS vyžaduje monitoring průměrně 17,1 pacientů (NNM = Number Needed to Monitor) bez stratifikace.

### 1.3 Cíl projektu

Vyvinout a validovat prediktivní model, který na základě rutinně dostupných klinických dat identifikuje pacienty s vysokou pravděpodobností záchytu FiS při holterovské monitoraci, a tím umožní efektivnější alokaci kapacit pro toto vyšetření.

Konkrétní přínos: Model snižuje NNM ze 17,1 na 12,8 (−25 %) při zachování senzitivity 95,5 % (21 z 22 případů FiS zachyceno). Ze 108 pacientů doporučených k vyloučení z monitoringu byl model v 99,1 % správný (NPV = 99,1 %).

---

## 2. Zainteresované strany (Stakeholders)

| Stakeholder | Role | Zájem |
|-------------|------|-------|
| Indikující lékaři (neurologové, kardiologové) | Primární uživatelé | Rychlá a přesná stratifikace pacientů |
| Tým holterovské monitorace (MDT Brno) | Příjemce výstupu stratifikace | Efektivní využití kapacit monitorace |
| Pacienti po iCMP/TIA | Koncoví příjemci péče | Včasný záchyt FiS a zahájení léčby |
| Vedení MDT | Administrativa | Nákladová efektivita, kvalita péče |
| TAČR | Financující instituce | Splnění výstupů projektu, přenositelnost výsledků |
| Vývojový tým (VUT / MDT) | Tvůrci systému | Reprodukovatelnost, open-source přístupnost |

---

## 3. Funkční požadavky

### FR1 — Přijetí vstupů a vrácení rizikové kategorie

Systém musí přijmout tři klinické vstupy a vrátit výsledek ve formě rizikové kategorie:

- **Vstup 1:** CHA₂DS₂-VASc skóre (celé číslo 0–9)
- **Vstup 2:** Přítomnost teritoriálního infarktu (binárně: ano/ne)
- **Vstup 3:** Hyperlipidémie v osobní anamnéze (binárně: ano/ne)
- **Výstup:** Riziková kategorie (Nízké / Střední / Vysoké / Velmi vysoké)

### FR2 — Kalibrovaná pravděpodobnost záchytu

Systém musí vrátit kalibrovanou pravděpodobnost záchytu FiS (hodnota 0–1) reflektující kohortovou prevalenci FiS ve STROCZE kohortu (~5,8 %). Pravděpodobnost musí být klinicky interpretovatelná — nesmí být uměle inflována vyvažováním tříd.

### FR3 — Relativní kategorizace k prevalenci

Rizikové kategorie jsou definovány jako násobky kohortové prevalence (5,84 %), nikoliv jako absolutní prahové hodnoty:
- Nízké: pravděpodobnost < 1× prevalence (< 5,84 %)
- Střední: 1–1,5× prevalence (5,84–8,76 %)
- Vysoké: 1,5–2× prevalence (8,76–11,68 %)
- Velmi vysoké: > 2× prevalence (> 11,68 %)

Tato definice je robustní vůči změnám prevalence v budoucích kohortách — při aktualizaci modelu stačí aktualizovat hodnotu prevalence.

### FR4 — Kalibrační kvalita

Brier skóre kalibrovaného modelu musí být ≤ Brier skóre no-skill baseline (predikce konstantní prevalence pro všechny pacienty). Pro prevalenci 5,84 % je no-skill baseline přibližně 0,055. Model musí dosáhnout Brier ≤ 0,055.

### FR5 — Webová dostupnost bez instalace

Kalkulátor musí být dostupný jako statická HTML stránka přístupná přes internetový prohlížeč bez nutnosti instalace jakéhokoli softwaru. Hosting musí být trvalý a bezplatný (GitHub Pages).

### FR6 — Open-source reprodukovatelnost

Veškerý zdrojový kód musí být veřejně dostupný v repozitáři GitHub. Spuštění tréninku modelu s fixním nastavením `random_state=42` musí vždy produkovat identické výsledky.

### FR7 — Statistická robustnost (EPV)

Events Per Variable (EPV) — počet výsledkových událostí na jeden prediktor modelu — musí být ≥ 10. Tato podmínka zajišťuje statistickou spolehlivost odhadnutých koeficientů logistické regrese. Při 89 pozitivních případech v trénovací sadě a 3 prediktorech je EPV = 29,7, což podmínku splňuje.

---

## 4. Nefunkční požadavky

### NFR1 — Diskriminační schopnost nad úrovní náhody

AUC-ROC modelu na testovacím souboru musí být statisticky signifikantně vyšší než 0,5 (náhodný klasifikátor). Toto musí být doloženo 95% konfidečním intervalem odhadnutým bootstrap metodou (1 000 iterací), přičemž dolní mez CI musí být > 0,5.

*Dosaženo:* AUC = 0,647, 95% CI [0,543–0,740].

### NFR2 — Bezpečnost rule-out rozhodnutí (NPV ≥ 95 %)

Záporná prediktivní hodnota (NPV) pro kategorii Nízké nesmí klesnout pod 95 %. Tato hodnota reprezentuje klinicky přijatelné riziko přehlédnutí FiS u pacientů doporučených k vyloučení z holterovské monitorace.

*Dosaženo:* NPV kategorie Nízké = 95,7 % při zvoleném prahu 1× prevalence.

### NFR3 — Interpretovatelnost pro klinické prostředí

Model musí být interpretovatelnný pro zdravotnické pracovníky prostřednictvím bodového systému (scorecard). Každý prediktor musí mít jasně definovaný koeficient a příspěvek v bodech na jednotku. Bodový systém umožňuje manuální výpočet nebo rychlé mentální odhadnutí rizika bez počítače.

### NFR4 — Responsivita pro mobilní zařízení

HTML kalkulačka musí být použitelná na mobilních zařízeních (telefon, tablet). Lékaři a sestry v nemocničním prostředí využívají mobilní zařízení jako primární přístupový bod k online nástrojům.

### NFR5 — Bezpečnost dat pacientů

Žádná individuální pacientská data nesmějí být součástí veřejného repozitáře GitHub. Model je reprezentován výhradně svými koeficienty uloženými v souboru `risk_scorecard.json`. Zdrojová data (složka `src/`) jsou trvale vyloučena ze sledování Gitem (`.gitignore`).

---

## 5. Případy použití

### UC1 — Stratifikace pacienta k holterovské monitoraci

**Aktér:** Neurolog / kardiolog  
**Předpoklad:** Pacient hospitalizovaný po iCMP nebo TIA  
**Tok událostí:**
1. Lékař zjistí hodnotu CHA₂DS₂-VASc, typ ischemie a anamnézu hyperlipidémie z dokumentace.
2. Lékař zadá hodnoty do HTML kalkulačky.
3. Kalkulačka vrátí rizikovou kategorii a pravděpodobnost.
4. Lékař využije výsledek jako podpůrnou informaci při rozhodování o indikaci holterovské monitorace.

**Výsledek:** Pacient je buď indikován k holterovské monitoraci (kategorie Střední–Velmi vysoké) nebo je sledován standardně (kategorie Nízké), vždy s ohledem na celkový klinický kontext.

### UC2 — Vyloučení nízkorizikového pacienta

**Aktér:** Neurolog / kardiolog  
**Předpoklad:** Pacient v kategorii Nízké  
**Tok událostí:**
1. Kalkulátor vrátí kategorii Nízké a pravděpodobnost < 5,84 %.
2. Lékař je informován, že NPV pro tuto kategorii je 95,7 % (při prahu 1× prevalence).
3. Lékař posoudí celkový klinický obraz a rozhodne o neindikaci holterovské monitorace.

**Výsledek:** Kapacita holterovské monitorace je ušetřena pro pacienty s vyšším rizikem.

### UC3 — Výzkumné využití modelu

**Aktér:** Výzkumný pracovník / student  
**Tok událostí:**
1. Výzkumník klonuje repozitář z GitHubu.
2. Spustí `python predict.py` pro demonstraci fungování modelu.
3. Nebo spustí `python risk_model.py` pro reprodukci celého tréninku.

**Výsledek:** Plná reprodukovatelnost výsledků, otevřenost ke srovnání s jinými modely.

---

## 6. Co systém nedělá (mimo rozsah)

Následující funkce jsou **explicitně mimo rozsah** tohoto projektu:

- **Diagnostika FiS:** Systém predikuje pravděpodobnost *záchytu* FiS při holterovské monitoraci, nikoliv přítomnost FiS jako takové.
- **Doporučení antikoagulační terapie:** Systém neurčuje, zda a jakou antikoagulační léčbu nasadit — to je výhradně rozhodnutí klinického lékaře.
- **Predikce jiných výsledků:** Systém nepredikuje mortalitu, riziko recidivy CMP, délku hospitalizace ani jiné klinické outcomes.
- **Zpracování zobrazovacích dat:** Systém nevyužívá CT ani MRI snímky.
- **Integrace do nemocničního informačního systému:** Systém je samostatná webová aplikace bez přímého napojení na NIS nebo EHR.

---

## 7. Akceptační kritéria

Systém je považován za funkční a připravený k odevzdání, pokud jsou splněna všechna následující kritéria:

| Kritérium | Cílová hodnota | Dosažená hodnota |
|-----------|---------------|-----------------|
| AUC-ROC (testovací sada) | > 0,5 (CI dolní mez) | 0,647 [0,543–0,740] ✓ |
| NPV kategorie Nízké | ≥ 95 % | 95,7 % ✓ |
| Brier skóre (kalibrovaný) | ≤ 0,055 | 0,0545 ✓ |
| EPV | ≥ 10 | 29,7 ✓ |
| HTML kalkulačka dostupná online | GitHub Pages | Splněno ✓ |
| Open-source repozitář | GitHub, veřejný | Splněno ✓ |
| TAČR dokumentace (5 dokumentů) | Úplná | Splněno ✓ |
| Zdrojová data MIMO repozitář | .gitignore | Splněno ✓ |

---

*Dokument připraven v rámci projektu NCK FEIM - DP001N (Healthcare). Verze 1.0, 2026-06-27.*
