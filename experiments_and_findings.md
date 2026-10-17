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

\[
E_{\mathrm{HF}}=-2.215215410025\ \mathrm{Ha}.
\]

Globální QCMPS pracuje s osmi fyzickými a zpravidla dvěma bond qubity. Před aplikací každého lokálního bloku se bond qubity pomocí SWAP bran posunou k následujícímu fyzickému qubitu. Pořadí fyzických qubitů přitom zůstává zachované.

Používaly se dvě hlavní účelové funkce:

1. Energetická minimalizace

   \[
   E(\boldsymbol\theta)
   =\langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle.
   \]

2. Maximalizace překryvu, implementovaná jako minimalizace infidelity

   \[
   \mathcal L_{\mathrm{ov}}(\boldsymbol\theta)
   =1-F(\boldsymbol\theta),
   \qquad
   F=|\langle\psi_{\mathrm{target}}|\psi(\boldsymbol\theta)\rangle|^2.
   \]

V obvodu se překryv měří připojením inverzního cílového obvodu a výpočtem pravděpodobnosti nulového stavu na fyzických qubitech.

## 1. Přímá optimalizace energie QCMPS z HF stavu

### Motivace

Prvním cílem bylo reprodukovat výsledky globálního QCMPS ansatzu z článku: začít v HF stavu, nastavit parametrické QCMPS bloky na identitu a přímo minimalizovat energii.

### Postup

- HF stav byl připraven pomocí `HFPreset`.
- Všechny QCMPS parametry byly nastaveny na nulu, takže parametrické brány byly identity.
- Optimalizátor COBYLA dostal přímo energetický objective.
- Byly zkoušeny různé počty evaluací a později také různé parametrizace bloků.

Současná podoba tohoto experimentu je v [`energy_optimization.py`](../experiments/energy_optimization.py).

### Pozorování

QCMPS inicializovaný přesně v identitě se od HF řešení prakticky nevzdálil. Energetická optimalizace vracela HF energii nebo změny na úrovni numerické chyby. Stejný jev se opakoval i po zvýšení dostupného počtu evaluací.

### Závěr

HF bod je pro používanou parametrizaci stacionární a je obklopen velmi plochou oblastí. Samotné zvýšení počtu COBYLA evaluací neřeší absenci užitečného lokálního směru. Problém proto není jen v předčasném ukončení optimalizátoru.

## 2. Náhodný šum kolem nulových parametrů

### Motivace

Pokud je gradient přesně v nule nulový, malé náhodné posunutí by mohlo odstranit symetrii a dát optimalizátoru nenulový směr.

### Postup

- QCMPS parametry byly inicializovány malým rovnoměrným šumem kolem nuly.
- Bylo vyzkoušeno několik velikostí šumu.
- Experiment se opakoval pro několik náhodných seedů.
- Následovala standardní energetická nebo překryvová optimalizace.

### Pozorování

Malý šum změnil stav i energii, ale optimalizátor se vracel do okolí HF větve nebo zůstal v téměř ekvivalentním bodě. Pro velmi malý krok, například kolem `0.001`, se překryv v prvních evaluacích téměř neměnil. Různé seedy nepřinesly systematicky lepší řešení.

### Závěr

Problém není způsoben jedinou nešťastnou volbou seedu. Malý izotropní šum většinou zasahuje ploché nebo redundantní parametrické směry. Větší šum sice opustí bezprostřední okolí identity, ale typicky zvýší energii a neposkytne spolehlivou cestu ke korelovanému minimu.

## 3. Zvýšení bond dimension

### Motivace

Mohlo jít o nedostatečnou expresivitu QCMPS s malým počtem bond qubitů. Proto byly provedeny běhy s větší bond dimension.

### Postup

- Zvýšil se počet bond qubitů a tím i velikost lokálních QCMPS bloků.
- QCMPS byl znovu inicializován v identitě nebo malým šumem.
- Optimalizace proběhla stejným způsobem jako pro menší bond dimension.

### Pozorování

Větší bond dimension sama o sobě nepřinesla únik z HF oblasti. Zvýšil se počet parametrů i cena jedné evaluace, ale lokální optimalizátor stále neměl spolehlivý směr k nižší energii.

### Závěr

Expresivita a optimalizovatelnost jsou odlišné vlastnosti. Větší ansatz může hledaný stav reprezentovat, ale současně vytváří více redundantních a plochých směrů. Zvyšování bond dimension bez lepší inicializace nebo parametrizace problém zhoršuje výpočetně.

## 4. Inicializace pomocí předpočítaného MPS/DMRG stavu

### Motivace

Článek o MPS pretrainingu motivoval použití klasicky získaného MPS jako počátečního bodu. Myšlenkou bylo převést DMRG tenzory na lokální unitární matice QCMPS a potom buď minimalizovat energii, nebo maximalizovat překryv s předtrénovaným stavem.

### Postup

Ve starší implementaci byly připraveny nástroje pro:

- načtení nebo výpočet MPS tenzorů,
- rozšíření MPS izometrií na unitární matice,
- vložení těchto matic do lokálních QCMPS bloků,
- použití výsledných matic jako počátečního bodu optimalizace.

Relevantní starší kód je zejména v:

- [`mps_initial_guess.py`](../old_code/qcmps_old_package/mps_initial_guess.py),
- [`qcmps_ansatz.py`](../old_code/qcmps_old_package/qcmps_ansatz.py),
- [`torch_ansatz.py`](../old_code/qcmps_old_package/torch_ansatz.py).

### Pozorování

Ani tato cesta v původních pokusech spolehlivě nereprodukovala výsledky článku. Jako důležitý zdroj problému se ukázalo pořadí spin-orbitalů a způsob, jakým jsou fyzické a bond indexy při převodu a swapování uspořádány. Další citlivost vzniká při doplnění izometrie na celou unitární matici: toto doplnění není jednoznačné a stejné MPS může vést k velmi rozdílným parametrům obvodu.

### Závěr

MPS inicializace je principiálně nejsilnější známá cesta, protože používá informaci o cílovém stavu. Není ale pouze otázkou vložení DMRG matic do bran. Je nutné přesně sjednotit orbital ordering, konvenci indexů MPS, orientaci bond indexů, endianitu qubitů a postup doplnění izometrie na unitární matici. Bez těchto kontrol nemusí dobrý MPS odpovídat dobrému bodu v parametrizaci obvodu.

## 5. Double-excitation inicializace následovaná QCMPS optimalizací

### Motivace

HF je lokální minimum vzhledem k jednoduchým změnám orbitalů, zatímco elektronová korelace se objevuje především přes double excitace. Proto vznikl samostatný `DoubleExcitation` komponent, který měl nejprve vytvořit korelovaný stav a až potom předat optimalizaci QCMPS.

### Postup

Experiment měl dvě fáze:

1. QCMPS byl fixován na identitu a optimalizovaly se pouze double-excitation parametry.
2. Optimalizované double excitace se zafixovaly a optimalizovaly se pouze parametry QCMPS.

COBYLA dostala pro první fázi přibližně 90 evaluací, aby se nehledalo přesné minimum double-excitation ansatzu, ale pouze bod mimo HF oblast.

### Pozorování

Double-excitation část skutečně snížila energii. V různých bězích dosáhla přibližně

\[
E_{\mathrm{DE}}\approx -2.2310\ \mathrm{Ha},
\]

například:

```text
HF:                    -2.215215410025 Ha
po DE optimalizaci:    -2.231061242321 Ha
```

Následná optimalizace QCMPS však energii dále nezlepšila. V kontrolním běhu zůstal výsledek přesně na

```text
-2.231061242321 Ha
```

po celou QCMPS fázi.

### Závěr

Double-excitation brána umí opustit HF minimum, ale QCMPS se z takto připraveného stavu nenaučil odpovídající korelovanou transformaci. Korelace zůstala v explicitní DE části. Tento experiment proto neřeší samostatnou inicializaci QCMPS; pouze ukazuje, že zvolený cílový korelovaný stav má nižší energii a je dosažitelný fyzikálně motivovaným operátorem.

## 6. Přímá maximalizace překryvu s double-excitation stavem

### Motivace

Energetický objective může být velmi plochý a mnoho různých stavů může mít podobnou energii. Přímý překryv s již připraveným korelovaným stavem poskytuje konkrétnější učící signál. Cílem bylo odstranit explicitní DE bránu z výsledného ansatzu a její účinek zkomprimovat do QCMPS.

### Postup

- Nejprve se optimalizoval cílový stav `HF → DoubleExcitation`.
- Inverzní cílový obvod se připojil za `HF → QCMPS`.
- Minimalizovala se pravděpodobnost, že fyzické qubity po inverzi nejsou v nulovém stavu.
- Byla použita COBYLA a také náhodná inicializace QCMPS parametrů.

Současná základní verze je v [`overlap_optimization.py`](../experiments/overlap_optimization.py).

### Pozorování

Jeden z charakteristických běhů s náhodnými parametry skončil:

```text
Initial/observed overlap: 0.975763140671
Final overlap:            0.975763140671
Evaluations:              5000
Success:                  False (MAXFUN)
```

Optimalizátor spotřeboval celý rozpočet, aniž by překryv viditelně zlepšil.

### Závěr

Záměna energetického objective za překryv sama o sobě problém nevyřešila. Překryv je fyzikálně vhodnější objective pro kompresi známého stavu, ale jeho krajina v používaných QCMPS souřadnicích zůstává velmi plochá a redundantní.

## 7. Postupný neboli step-overlap objective

### Motivace

Přímý přechod od identity k celému korelovanému cíli může být příliš obtížný. Proto byl zaveden continuation postup, ve kterém se QCMPS učí stále větší část transformace po malých krocích.

### Konstrukce objective

Označme optimalizované DE parametry jako \(\boldsymbol\theta^*\) a continuation parametr jako \(s\in[0,1]\). Optimalizovaný stav má tvar

\[
|\psi_s(\boldsymbol\phi)\rangle
=U_{\mathrm{QCMPS}}(\boldsymbol\phi)
 U_{\mathrm{DE}}((1-s)\boldsymbol\theta^*)
 |\mathrm{HF}\rangle.
\]

Cílem zůstává plný stav

\[
|\psi_{\mathrm{target}}\rangle
=U_{\mathrm{DE}}(\boldsymbol\theta^*)|\mathrm{HF}\rangle.
\]

Pro \(s=0\) je řešení známé: QCMPS je identita a DE obvod připravuje celý cíl. Při zvětšování \(s\) se DE transformace zeslabuje a QCMPS by ji měl postupně převzít. Pro \(s=1\) je DE část identita a celý cílový stav musí připravit QCMPS.

Implementace je v [`step_overlap_opt.py`](../experiments/step_overlap_opt.py).

### Pozorování

- Pro \(s=0\) byl překryv přesně 1.
- Už pro velmi malé nenulové \(s\) se optimalizace měnila jen minimálně.
- Při postupném zvyšování \(s\) se překryv zhoršoval přibližně s velikostí přesouvané transformace.
- Menší continuation kroky byly stabilnější než skok přímo na velké \(s\), ale QCMPS zůstával na větvi blízké identitě.
- Změna COBYLA `rhobeg` ovlivnila velikost prvních kroků, neodstranila však základní plochost.

### Závěr

Continuation metoda poskytla lépe kontrolovaný experiment, ale nevytvořila nový užitečný směr v parametrickém prostoru. Optimalizátor převážně sledoval identitní/HF větev místo toho, aby přenesl korelaci z DE obvodu do QCMPS.

## 8. Analytický gradient a Hessian step-overlap objective

### Motivace

Bylo potřeba rozlišit, zda COBYLA pouze volí příliš malé kroky, nebo zda je objective kolem identity skutečně lokálně plochý.

### Postup

- Pro malý continuation krok se spočítal gradient překryvového objective v okolí nulových QCMPS parametrů.
- Pro \(s=0.05\) byl numericky analyzován Hessian.
- Zkoumala se také hodnost tečného prostoru stavů vytvořeného jednotlivými parametrickými derivacemi.

### Pozorování

- Gradient v nulovém bodě byl nulový nebo numericky nerozlišitelný od nuly.
- Hessian byl silně anisotropní a obsahoval velké množství nulových nebo téměř nulových směrů.
- U 192parametrové identity-centered parametrizace měl stavový tečný prostor na HF pouze přibližně hodnost 19. Přibližně 173 parametrických směrů tedy bylo v prvním řádu redundantních nebo neviditelných.
- Předchozí parametrizace měla ještě nižší efektivní hodnost, přibližně 10.

### Interpretace

Parametrický prostor není v okolí HF podobný běžné hladké misce s jedním dobře definovaným směrem dolů. Obsahuje rozsáhlé ploché vrstvy a mnoho různých parametrických bodů reprezentuje stejný nebo téměř stejný fyzikální stav. Optimalizátor může výrazně změnit normu parametrů, aniž by významně změnil stav nebo energii.

### Závěr

Pozorované chování není primárně chyba COBYLA. Je důsledkem lokální geometrie zvolené parametrizace. Pro únik je potřeba parametrizace s fyzikálně relevantnějšími tangentními směry, koordinovaný vícerozměrný krok nebo inicializace mimo HF větev.

## 9. Odpudivá regularizace parametrů

### Motivace

Standardní L2 regularizace tlačí parametry k nule. Zde byl vyzkoušen opačný princip: přidat člen, který penalizuje nulové parametry a vytlačí optimalizaci z HF bodu.

Použitý objective měl tvar

\[
\mathcal L_{\mathrm{reg}}(\boldsymbol\theta)
=1-F(\boldsymbol\theta)
+\alpha
\frac{1}
{\varepsilon+\lambda\sum_i\sin^2\theta_i}.
\]

Parametry byly inicializovány malým šumem, aby regularizační člen měl definovaný směr. Síla \(\alpha\) se v průběhu continuation kroků zmenšovala. Po každé regularizované fázi následovala čistá optimalizace překryvu.

Implementace a dochovaný log jsou v:

- [`step_overlap_with_reg.py`](../experiments/step_overlap_with_reg.py),
- [`step_overlap_with_reg_log.txt`](../experiments/step_overlap_with_reg_log.txt).

### Pozorování

Regularizace dokázala zvětšit normu parametrů a dostat optimalizátor z přesného nulového bodu. V prvních malých continuation krocích po následné čisté optimalizaci vznikala drobná zlepšení překryvu. Zlepšení však byla velmi malá a s rostoucím \(s\) se prakticky vytratila.

V dochovaném jemném běhu byly zaznamenány například hodnoty:

| \(s\) | konečný překryv |
|---:|---:|
| 0.05 | 0.999955981907 |
| 0.25 | 0.998981192490 |
| 0.50 | 0.995927950404 |
| 0.75 | 0.990846326931 |
| 1.00 | 0.983756748481 |

Na konci běhu:

```text
Final overlap:       0.983756748481
QCMPS energy:       -2.215215405354 Ha
HF energy:          -2.215215410025 Ha
```

Výsledná QCMPS energie byla stále nepatrně vyšší než HF energie. Rozdíl byl pouze několik \(10^{-9}\) Ha, tedy fyzikálně žádné získání korelační energie.

Hrubší varianta pouze s velkými continuation kroky skončila kolem překryvu `0.9829236`, takže zjemnění kroků pomohlo jen mírně.

### Závěr

Regularizace úspěšně hýbe parametry, ale neumí rozlišit fyzikálně účinný směr od gauge nebo redundantního směru. Optimalizátor proto může splnit tlak „pryč od nuly“, aniž by QCMPS převzal korelovanou část cílového stavu. Parametrická vzdálenost od HF není spolehlivou mírou fyzikální vzdálenosti stavů.

## 10. Změna parametrizace elementárních bran

### Motivace

Původní brány měly kolem identity mnoho neaktivních nebo redundantních parametrů. Zkoumalo se proto, zda změna generátorů zlepší lokální geometrii.

### Testované varianty

1. Obecná jednoqubitová `U` a controlled-`U` parametrizace.
2. Identity-centered varianta

   \[
   R_X(\alpha)R_Y(\beta)R_Z(\gamma)
   \]

   a

   \[
   R_{XX}(\alpha)R_{YY}(\beta)R_{ZZ}(\gamma).
   \]

3. Jednoparametrová varianta

   \[
   R_Y(\theta),\qquad R_{XX}(\phi).
   \]

### Pozorování

Identity-centered tříparametrové brány zvýšily hodnost stavového tečného prostoru kolem HF, ale velká většina ze 192 parametrů zůstala v prvním řádu redundantní. Jednoparametrové brány snížily velikost ansatzu ze 192 na 64 parametrů:

```text
3 qubity v lokálním bloku:
RY vrstva  = 3 parametry
RXX vrstva = 2 parametry
RY vrstva  = 3 parametry
celkem     = 8 parametrů na fyzický qubit

8 fyzických qubitů × 8 = 64 parametrů
```

Nulové parametry stále přesně reprezentují HF bod. Jednoparametrová varianta je levnější a odstraňuje zjevnou lokální redundanci mezi `RX`/`RY` a `RXX`/`RYY` i čistě fázové směry `RZ`/`RZZ` na výpočetním bazickém stavu.

### Závěr

Redukce parametrů zlepšuje interpretovatelnost, cenu evaluace a numerickou podmíněnost. Dosavadní regularizovaný běh ale ukázal, že ani tato změna sama o sobě neodstranila HF větev. Jde o užitečné zjednodušení problému, nikoliv o úplné řešení optimalizace.

## Celkové poznatky

1. **HF není problém pouze pro jeden konkrétní optimalizátor.** Přímá energie, přímý překryv i continuation objective vykazují kolem identity velmi slabý učící signál.

2. **Více parametrů automaticky nepomáhá.** Vyšší bond dimension a obecnější brány zvyšují expresivitu, ale také počet plochých a redundantních směrů.

3. **Double excitace potvrzují existenci dostupného korelovaného stavu.** Samostatný DE ansatz energii sníží přibližně o 15.8 mHa, QCMPS však tuto transformaci nedokázal převzít.

4. **Překryv je vhodnější diagnostika než samotná energie, ale potřebuje dobrou parametrizaci.** Přímá maximalizace překryvu se bez vhodného počátečního bodu také zastavila.

5. **Malý šum a více seedů neřeší strukturální degeneraci.** Pomáhají pouze tehdy, když v okolí existují dostupné užitečné směry.

6. **Parametrická norma neměří změnu fyzikálního stavu.** Odpudivá regularizace dokázala zvětšit parametry, ale velká část pohybu proběhla v redundantních směrech.

7. **Orbital ordering a práce s bond/physical indexy jsou kritické.** Chyba v jejich pořadí může znehodnotit MPS inicializaci i jinak správný překryvový experiment.

8. **Nejslibnější cesta zůstává informovaná inicializace nebo fyzikálně motivované koordinované směry.** Patří sem korektní převod MPS izometrií, excitacemi řízená parametrizace, bloková optimalizace nebo práce v nízkorozměrném prostoru určeném Jacobianem či Hessianem.

## Aktuální stav implementace

Současná struktura odděluje:

- parametrizované elementární brány,
- způsob jejich rozmístění v `LayerType`,
- pořadí vrstev dodané uživatelem,
- globální QCMPS sweep se SWAP operacemi.

Díky tomu lze jednotlivé hypotézy testovat bez změny `GlobalQCMPS`, například:

```python
u_layer = SingleQubitLayerType(UGate)
entangling_layer = EntanglingLayerType(CUGate)

qcmps = GlobalQCMPS(
    n_phys_qubits,
    n_bond_qubits,
    [u_layer, entangling_layer, u_layer],
)
```

Regularizovaný experiment používá redukovanou `RY–RXX–RY` parametrizaci se 64 parametry. Ostatní současné experimenty používají obecnější `U–CU–U` variantu se 192 parametry. Toto rozdělení je záměrné a umožňuje odděleně porovnat expresivitu a optimalizovatelnost.
