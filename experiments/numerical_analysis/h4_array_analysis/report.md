# Analýza H4 QCMPS array experimentu

## Reference

- Celková HF energie: `-2.215215410025 Ha`
- Celková CASCI energie: `-2.255962073366 Ha`
- Korelační energie: `40.746663 mHa`
- Počet analyzovaných běhů: `24`

## Nejlepší výsledek

Nejlepší výsledek array experimentu má `q=2` (`D=4`), seed `0`, `4` vrstvy a `384` parametrů.

- Celková energie: `-2.228225343638 Ha`
- Chyba vůči CASCI: `27.736730 mHa`
- Získaná korelační energie: `31.929 %`

## Porovnání čtyřvrstvých ansatzů

| Bond qubity | Bond dimension | Průměrná energie [Ha] | Směrodatná odchylka seedů [mHa] | Nejlepší energie [Ha] | Nejlepší seed | Nejlepší korelace [%] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | -2.221242796484 | 0.969116 | -2.222211912685 | 1 | 17.171 |
| 2 | 4 | -2.227587808522 | 0.637535 | -2.228225343638 | 0 | 31.929 |
| 3 | 8 | -2.218278780647 | 3.059940 | -2.221338721039 | 1 | 15.028 |

Byly použity pouze dva seedy, takže průměr a směrodatná odchylka jsou jen popisné hodnoty, nikoliv spolehlivý statistický odhad.

## Celé trajektorie od jedné do čtyř vrstev

| Bond qubity | Bond dimension | Seed | Zisk L1 → L4 [mHa] | Finální chyba [mHa] | Finální korelace [%] | Kumulativní evaluace |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0 | 0.002097 | 35.688393 | 12.414 | 15795 |
| 1 | 2 | 1 | 0.022787 | 33.750161 | 17.171 | 15950 |
| 2 | 4 | 0 | 13.009307 | 27.736730 | 31.929 | 103532 |
| 2 | 4 | 1 | 0.160063 | 29.011800 | 28.800 | 53593 |
| 3 | 8 | 0 | 0.002473 | 40.743233 | 0.008 | 41633 |
| 3 | 8 | 1 | 6.115894 | 34.623352 | 15.028 | 130129 |

## Diagnostika konvergence

- Globální etapy ukončené neúspěchem nebo limitem: `11/24`
- Sweep etapy ukončené maximálním počtem sweepů: `6/18`
- `9` globálních běhů: STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT
- `2` globálních běhů: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Produktivní trajektorie často skončily na limitu evaluací nebo sweepů. Několik formálně zkonvergovaných běhů naopak zůstalo blízko HF. Příznak úspěchu optimalizátoru proto sám o sobě nevypovídá o fyzikální kvalitě řešení.

## Hlavní závěry

1. Výsledek silně závisí na seedu. Stejně velký ansatz může zůstat u HF, nebo získat významnou část korelační energie.
2. Vyšší bond dimension nevede monotónně k lepší optimalizované energii. Větší parametrické prostory se současným numerickým gradientem optimalizují hůře.
3. Přidávání vrstev pomáhá pouze podél produktivních optimalizačních trajektorií. Jiné seedy stagnují téměř okamžitě navzdory vyšší expresivitě.
4. Nejlepší array běh má stále velkou chybu vůči CASCI. Z těchto dat nelze tvrdit, že je zkonvergovaná expresivita ani optimalizace.
5. Procenta korelační energie jsou předběžná, dokud nebude u SO(4) ansatzu ověřena střední hodnota a variance částicového čísla.

## Vygenerované soubory

- `runs.csv`: jeden řádek pro každý seed, bond dimension a počet vrstev.
- `aggregated.csv`: průměr, rozptyl a nejlepší výsledek pro každou bond dimension a počet vrstev.
- `trajectories.csv`: souhrn celé adaptivní trajektorie každého seedu.
- `overview.png`: grafy energie, korelační energie, přesnosti a ceny optimalizace.
