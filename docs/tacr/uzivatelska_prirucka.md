# Uživatelská příručka
## Kalkulátor rizika záchytu fibrilace síní po ischemické cévní mozkové příhodě

---

**Projekt:** NCK FEIM - DP001N (Healthcare) — Identifikace rizikových faktorů fibrilace síní  
**Verze dokumentu:** 1.0  
**Datum:** 2026-06-27  
**Autoři:** Valentýna Provazník, Veronika Bulková, Richard Ředina  
**Instituce:** Vysoké učení technické v Brně / Medical Data Transfer s.r.o.

---

## 1. Účel nástroje

Tento nástroj slouží k odhadu pravděpodobnosti záchytu fibrilace síní (FiS) u pacientů po ischemické cévní mozkové příhodě (iCMP) nebo tranzitorní ischemické atace (TIA), u nichž je indikována dlouhodobá holterovská monitorace.

Dlouhodobý monitoring je metoda pro ziskání neinvazivních záznamů srdeční aktivity (Holter EKG, typicky 24–48 hodin). Záchyt fibrilace síní u pacientů po CMP je klinicky zásadní: FiS je příčinou přibližně 20–30 % ischemických CMP a u pacientů, u nichž je FiS zachycena, je indikována antikoagulační léčba ke snížení rizika recidivy.

Kalkulátor **nestanovuje diagnózu FiS** — určuje pravděpodobnost, že FiS bude zachycena při dlouhodobém monitoringu. Je určen jako **podpůrný nástroj pro stratifikaci pacientů**, nikoliv jako náhrada klinického rozhodnutí.

---

## 2. Určení nástroje

**Pro koho je nástroj určen:**
- Neurologové a kardiologové péče o pacienty po iCMP/TIA
- Kliničtí pracovníci indikující dlouhodobý monitoring
- Zdravotničtí pracovníci provádějící stratifikaci pacientů k vyšetření

**Pro koho nástroj určen není:**
- Pacienti mimo kohortu iCMP/TIA (jiné indikace k Holter EKG)
- Autonomní diagnostika bez odborného dohledu
- Populace mimo českou klinickou praxi (validace pouze na STROCZE kohortě)

---

## 3. Vstupní proměnné

Kalkulátor vyžaduje tři klinické údaje, které jsou standardně dostupné z dokumentace pacienta po iCMP/TIA.

### 3.1 CHA₂DS₂-VASc skóre

Celé číslo v rozsahu 0–9. CHA₂DS₂-VASc je standardizované skóre pro odhad rizika tromboembolické příhody u pacientů s fibrilací síní:

| Písmeno | Klinický faktor | Body |
|---------|----------------|------|
| C | Kongestivní srdeční selhání | 1 |
| H | Hypertenze | 1 |
| A₂ | Věk ≥ 75 let | 2 |
| D | Diabetes mellitus | 1 |
| S₂ | Předchozí CMP nebo TIA | 2 |
| V | Vaskulární onemocnění | 1 |
| A | Věk 65–74 let | 1 |
| Sc | Pohlaví (žena) | 1 |

Hodnotu CHA₂DS₂-VASc zpravidla najdete ve zdravotnické dokumentaci nebo ji lze jednoduše spočítat sečtením přítomných faktorů. Skóre je počítáno při přijetí nebo v průběhu hospitalizace.

### 3.2 Typ akutní ischemie — Teritoriální infarkt

Binární proměnná (ano / ne). Teritoriální infarkt je velký kortikální nebo kortiko-subkortikální infarkt postihující celý nebo podstatnou část povodí mozkové tepny (nejčastěji a. cerebri media nebo a. cerebri anterior). Opakem je lakunární infarkt (malý, hluboký, subkortikální).

**Jak zjistit:** Typ ischemie je uveden v propouštěcí zprávě nebo v záznamu zobrazovací metody (CT, MRI mozku).

### 3.3 Hyperlipidémie v osobní anamnéze

Binární proměnná (ano / ne). Anamnesticky diagnostikovaná hyperlipidémie (dyslipidemie), nezávisle na aktuální medikaci.

**Upozornění — léčebný paradox:** Model přiřazuje hyperlipidémii **záporný** koeficient, tj. pacienti s hyperlipidémií mají v modelu *nižší* predikované riziko. Toto je pravděpodobně důsledek toho, že pacienti s diagnostikovanou hyperlipidémií jsou ve většině případů léčeni statiny, které snižují riziko FiS. Data STROCZE neobsahují informaci o konkrétní medikaci — tato interpretace je hypotézou, která nemůže být přímo ověřena v dostupných datech. Výsledek kalkulátoru u pacientů s hyperlipidémií interpretujte s ohledem na toto omezení.

---

## 4. Jak používat kalkulátor

### 4.1 HTML kalkulačka (doporučeno)

1. Otevřete v internetovém prohlížeči adresu: `https://ricRedi.github.io/afib-risk-strocze/`
2. Zadejte hodnotu CHA₂DS₂-VASc (celé číslo 0–9).
3. Vyberte typ akutní ischemie: **Teritoriální** nebo **Jiný / lakunární / neznámý**.
4. Vyberte, zda má pacient hyperlipidémii v anamnéze: **Ano** nebo **Ne / neznámo**.
5. Klikněte na tlačítko **Vypočítat riziko FiS**.
6. Výsledek se zobrazí ihned — riziková kategorie, kalibrovaná pravděpodobnost a poměr k prevalenci kohorty.

Kalkulačka funguje i na mobilních zařízeních (telefon, tablet) bez nutnosti instalace.

### 4.2 Python demo (pro technické uživatele)

```bash
# Klonovat repozitář a nainstalovat závislosti
git clone https://github.com/RicRedi/afib-risk-strocze.git
cd afib-risk-strocze
pip install -r requirements.txt

# Spustit demo
python predict.py
```

Demo zobrazí ukázkové predikce pro tři illustrativní pacienty s výpisem všech klinických vstupů, bodového skóre a rizikové kategorie.

---

## 5. Interpretace výsledku

Kalkulátor vrací **rizikovou kategorii** definovanou jako násobek kohortové prevalence FiS (5,8 % v kohortu STROCZE). Výstupem není přímá pravděpodobnost diagnózy FiS, ale odhad pravděpodobnosti jejího **záchytu při holterovské monitoraci**.

### 5.1 Čtyři rizikové kategorie

| Kategorie | Práh (násobek prevalence) | Odhadovaná pravděp. záchytu | Doporučení |
|-----------|--------------------------|----------------------------|------------|
| **Nízké** | < 1× prevalence (< 5,8 %) | < 5,8 % | Standardní sledování; holterovská monitorace s nízkou prioritou |
| **Střední** | 1–1,5× prevalence (5,8–8,7 %) | 5,8–8,7 % | Mírně zvýšené riziko; zvažte holterovskou monitoraci |
| **Vysoké** | 1,5–2× prevalence (8,7–11,7 %) | 8,7–11,7 % | Výrazně zvýšené riziko; holterovská monitorace doporučena |
| **Velmi vysoké** | > 2× prevalence (> 11,7 %) | > 11,7 % | Silné doporučení k holterovské monitoraci |

### 5.2 Zvláštní hodnota kategorie Nízké

Pacienti zařazení do kategorie **Nízké** (tj. predikovaná pravděpodobnost záchytu FiS nižší než kohortová prevalence) mají zápornou prediktivní hodnotu (NPV) **95,7 %** při zvoleném prahu 1× prevalence. To znamená, že ze 100 pacientů zařazených do kategorie Nízké nebude FiS zachycena u přibližně 95–96 z nich.

Na úrovni celého testovacího souboru (zahrnuje všechny pacienty predikované jako „pravděpodobně FiS−") dosahuje NPV modelu **99,1 %** — z 108 pacientů doporučených k vyloučení z monitoringu byl u 107 výsledek monitorace správně negativní a pouze u 1 pacienta (0,9 %) byla FiS přehlédnuta.

### 5.3 Příklady interpretace

**Příklad A — nízké riziko:**  
Pacient, 55 let, lakunární infarkt, CHA₂DS₂-VASc = 1, bez hyperlipidémie.  
→ Model predikuje kategori Nízké (pravděpodobnost ~4 %, 0,7× prevalence).  
→ Interpretace: FiS nepravděpodobná; standardní sledování odpovídá klinickému kontextu.

**Příklad B — vysoké riziko:**  
Pacient, 72 let, teritoriální infarkt, CHA₂DS₂-VASc = 6, hyperlipidémie v anamnéze.  
→ Model predikuje kategorii Velmi vysoké (pravděpodobnost ~11,8 %, 2,0× prevalence).  
→ Interpretace: Výrazně zvýšené riziko záchytu FiS; holterovská monitorace je silně indikována.

---

## 6. Limitace a kontraindikace

1. **Pouze pro pacienty po iCMP/TIA:** Kalkulátor byl vyvinut a validován výhradně na pacientech sledovaných v STROCZE kohortě (Czech Stroke Research database, MDT Brno). Použití u jiných populací (primární prevence, kardiologické diagnózy mimo CMP) není ověřeno.

2. **Klinické rozhodnutí zůstává na lékaři:** Výstup kalkulátoru je podpůrná informace. Konečné rozhodnutí o indikaci holterovské monitorace nebo antikoagulační léčby je výhradně v kompetenci ošetřujícího lékaře.

3. **Interní validace:** Model byl validován na interním testovacím souboru z téže kohorty (20 % dat, n = 377). Nezávislá prospektivní validace na nových pacientech nebyla provedena. Před rutinním klinickým nasazením je prospektivní validace nezbytná.

4. **Léčebný paradox hyperlipidémie:** Záporný koeficient hyperlipidémie je interpretován jako důsledek statinové léčby (viz oddíl 3.3). Tato hypotéza nemůže být v datech STROCZE ověřena z důvodu chybějící informace o medikaci.

5. **Nízká specifita:** Model byl optimalizován pro maximální senzitivitu (záchyt AFib+) na úkor specifity (30 %). Z 248 pacientů predikovaných jako FiS+ bude většina při holterovské monitoraci negativní. Toto je vědomý kompromis: model je navržen jako screeningový nástroj pro *vyloučení* nízkorizikových pacientů, nikoliv jako potvrzující diagnostický test.

---

## 7. Kontakt

**Valentýna Provazník**  
E-mail: provaznik@vut.cz  
Vysoké učení technické v Brně  

**Veronika Bulková**  
E-mail: veronika.bulkova@gmail.com  
Medical Data Transfer s.r.o.

**Richard Ředina**  
E-mail: 195715@vut.cz  
Vysoké učení technické v Brně  

---

*Tento nástroj je výzkumným prototypem financovaným projektem NCK FEIM - DP001N (Healthcare). Není schválen pro samostatné klinické rozhodování.*
