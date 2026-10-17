# QCMPS: přehled experimentů a dosavadních poznatků

## Společné nastavení současných experimentů

Hlavním testovaným systémem je lineární H4 načítaný z `fcidumps/H4_FCIDUMP`. Hamiltonián je po načtení permutován jako

```python
[0, 2, 4, 6, 1, 3, 5, 7]
```

Tím se pořadí spin-orbitalů přizpůsobí pořadí qubitů používanému ansatzem. Hartreeův–Fockův stav má obsazené qubity

```python
[0, 1, 2, 3]
```

a jeho celková energie v používaném Hamiltoniánu je

$$
E_{\mathrm{HF}}=-2.215215410025\ \mathrm{Ha}.
$$

Všechny testy jsou uloženy v složce `experiments`.

## 1. Přímá optimalizace energie QCMPS z HF stavu

Experiment je uveden v `energy_optimization.py`. Samotná energie QCMPS

$$
E(\boldsymbol\theta)
=\langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle
$$

byla optimalizována z HF inicializace (pomocí `HFPreset`) nebo z ansatzu s dvojitými excitacemi, který byl nejprve optimalizován nezávisle a následně použit jako vstup do QCMPS (`DoubleExcitation`).

### Pozorování

- Double excitace sama o sobě snížila energii velmi rychle přibližně na −2.2310 Ha.
- QCMPS se však nedokáže odpoutat od nulových parametrů $\boldsymbol\theta$ a zůstává přibližně v identitě.
- Nulový bod parametrů tvoří lokální minimum.
- Jednotlivé orbitaly spolu při malých změnách účinně nekomunikují, a proto tyto změny vedou mimo fyzikálně relevantní podprostor. Ansatz totiž nezachovává počet elektronů.
- Například pro vznik dvojité excitace se musí najednou koordinovaně posunout několik parametrů ve vhodném směru a o určitou vzdálenost. Takový pohyb COBYLA prakticky nedokáže najít.
- Při malém šumu se parametry okamžitě vracejí zpět, zatímco velký šum odpovídá startu s náhodnými parametry.
- Zvýšení bond dimension nebo počtu parametrů činí ansatz expresivnějším, ale nezlepšuje cestu k výsledku. Praktickou optimalizaci naopak dále komplikuje zvýšením počtu plochých směrů.

## 2. Maximalizace překryvu

Experiment je v souboru `overlap_optimization.py`. Myšlenka vychází z toho, že energetická účelová funkce může kvůli nezachovávání počtu částic vytvářet různá lokální minima. Jiná účelová funkce se proto může chovat odlišně.

Nejprve byla opět optimalizována vrstva `DoubleExcitation`. Následně se sestavil obvod s QCMPS, za který byl připojen invertovaný předtrénovaný blok `DoubleExcitation`. Minimalizovala se pravděpodobnost, že fyzické qubity po inverzi nejsou v nulovém stavu. Cílem bylo přenést informaci z `DoubleExcitation` do QCMPS.

### Pozorování

- Optimalizátor vždy využil celý počet iterací, aniž by se výsledek zlepšil.
- Záměna účelové funkce nepomohla. Překryv je pravděpodobně vhodnější účelová funkce, ale nulové parametry $\boldsymbol\theta$ celý proces komplikují.

## 3. Step-overlap účelová funkce

Experiment je uložen v `step_overlap_opt.py`. Motivací je skutečnost, že přímý přechod může být pro QCMPS příliš velkým skokem. Správná oblast parametrického prostoru se může nacházet daleko od nulových parametrů. Proto byl zaveden continuation postup, při kterém se QCMPS v každém kroku učí stále větší část transformace.

Označme optimalizované DE parametry jako $\boldsymbol\theta^*$ a continuation parametr jako $s\in[0,1]$. Optimalizovaný stav má tvar

$$
|\psi_s(\boldsymbol\phi)\rangle
=U_{\mathrm{QCMPS}}(\boldsymbol\phi)
 U_{\mathrm{DE}}((1-s)\boldsymbol\theta^*)
 |\mathrm{HF}\rangle.
$$

Cílem zůstává plný stav

$$
|\psi_{\mathrm{target}}\rangle
=U_{\mathrm{DE}}(\boldsymbol\theta^*)|\mathrm{HF}\rangle.
$$

Pro $s=0$ je řešení známé: QCMPS je identita a DE obvod připravuje celý cíl. Při zvětšování $s$ se DE transformace zeslabuje a QCMPS by ji měl postupně převzít. Pro $s=1$ je DE část identita a celý cílový stav musí připravit QCMPS.

### Pozorování

- Už pro velmi malé nenulové $s$ se optimalizace měnila jen minimálně.
- Při postupném zvyšování $s$ se překryv zhoršoval přibližně s velikostí přesouvané transformace.
- Menší continuation kroky byly stabilnější než skok přímo na velké $s$, ale QCMPS zůstával na větvi blízké identitě.
- Problém tedy přetrvává; zásadní překážkou zůstává plochost parametrického prostoru.
- Nulové parametry totiž generují redundantní směry.
- Původní ansatz z článku, který využívá Eulerovy úhly, není parametrizován jednoznačně, podobně jako zeměpisné nebo sférické souřadnice.
- Nacházíme-li se na pólu, který zde odpovídá nulovým parametrům, nezáleží na tom, zda měníme souřadnici směrem na východ, nebo na západ: stav zůstává stejný.
- Toto lokální minimum je proto roztažené přes velké množství dimenzí. Příslušné parametry jsou redundantní a jejich změna fyzikálně nemění stav. V neredundantních směrech však stále jde o lokální minimum, takže gradienty jsou v nule nulové.
- Mimo nulový bod už tato plochost prostoru neplatí, ale inicializace malým šumem vede rychle zpět k nulovým parametrům.
- Reparametrizace odstraňující ploché směry teoreticky funguje, ale stále neřeší problém lokálního minima.
- Byl spočten také Hessian, který potvrdil, že skutečně jde o lokální minimum, nikoli například o sedlový bod.

## 4. Odpudivá regularizace parametrů

Experiment je uložen v `step_overlap_with_reg.py`. Motivace pochází z machine learningu. Standardní L2 regularizace tlačí parametry k nule, čímž omezuje overfitting. Zde byl vyzkoušen opačný princip: přidat člen, který penalizuje nulové parametry a vytlačí optimalizaci z HF bodu.

Použitá účelová funkce měla tvar

$$
\mathcal L_{\mathrm{reg}}(\boldsymbol\theta)
=1-F(\boldsymbol\theta)
+\alpha
\frac{1}
{\varepsilon+\lambda\sum_i\sin^2\theta_i}.
$$

Parametry byly inicializovány malým šumem, aby regularizační člen měl definovaný směr. Síla $\alpha$ se v průběhu continuation kroků zmenšovala. Po každé regularizované fázi následovala čistá optimalizace překryvu.

Jeden dochovaný log je také v `step_overlap_with_reg_log.txt`.

### Pozorování

- Regularizace dokázala zvětšit normu parametrů a dostat optimalizátor z přesného nulového bodu. V prvních malých continuation krocích po následné čisté optimalizaci vznikala drobná zlepšení překryvu. Zlepšení však byla velmi malá a s rostoucím $s$ se prakticky vytratila.
- Regularizace úspěšně mění parametry, ale neumí rozlišit fyzikálně účinný směr od gauge nebo redundantního směru.
- Optimalizátor proto může splnit tlak pryč od nuly, aniž by QCMPS převzal korelovanou část cílového stavu. Parametrická vzdálenost od HF není spolehlivou mírou fyzikální vzdálenosti stavů.
- Parametry jsou z nuly vytlačovány především v plochých směrech, aniž by se samotný fyzikální stav změnil.
- Nahrazení původní parametrizace jinou vedlo k mírně rychlejší konvergenci, ale výslednou hodnotu účelové funkce prakticky neovlivnilo. Některé ploché směry se odstranily, lokální minimum však zůstalo zachováno.

## Celkové poznatky

1. **Obtíže v okolí HF nejsou spojeny pouze s jedním konkrétním optimalizátorem.** Přímá minimalizace energie, přímá maximalizace překryvu i continuation účelová funkce vykazují kolem identity velmi slabý optimalizační signál.

2. **Více parametrů automaticky nepomáhá.** Vyšší bond dimension a obecnější brány zvyšují expresivitu, ale také počet plochých a redundantních směrů.

3. **Double excitace potvrzují existenci dostupného korelovaného stavu.** Samostatný DE ansatz energii sníží přibližně o 15,8 mHa, QCMPS však tuto transformaci nedokázal převzít.

4. **Překryv je vhodnější diagnostika než samotná energie, ale potřebuje dobrou parametrizaci.** Přímá maximalizace překryvu se bez vhodného počátečního bodu také zastavila.

5. **Malý šum a více seedů neřeší strukturální degeneraci.** Pomáhají pouze tehdy, když v okolí existují dostupné užitečné směry.

6. **Parametrická norma neměří změnu fyzikálního stavu.** Odpudivá regularizace dokázala zvětšit parametry, ale velká část pohybu proběhla v redundantních směrech.

