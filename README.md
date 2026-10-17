# QCMPS

Experimentální implementace globálního QCMPS ansatzu v Qiskitu. Balíček umožňuje sestavit ansatz z menších parametrizovaných komponent, načíst molekulární Hamiltonián ze souboru FCIDUMP a optimalizovat energii nebo překryv.

## Instalace

Projekt vyžaduje Python 3.11 nebo novější. V kořenovém adresáři repozitáře spusťte:

```bash
python -m pip install -e .
```

Volba `-e` nainstaluje balíček v editovatelném režimu, takže se změny ve zdrojovém kódu projeví bez nové instalace.

## Základní použití

Následující příklad načte systém H4, připraví Hartreeův–Fockův stav, přidá QCMPS a optimalizuje jeho parametry pomocí COBYLA:

```python
from pathlib import Path

from scipy.optimize import minimize

from qcmps.ansatz import Ansatz, GlobalQCMPS, HFPreset
from qcmps.circuit_blocks import (
    CUGate,
    UGate,
    EntanglingLayerType,
    SingleQubitLayerType,
)
from qcmps.loader import FCIDumpLoader


# 1. Načtení Hamiltoniánu
loader = FCIDumpLoader(Path("fcidumps/H4_FCIDUMP"))
loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])

# 2. Rozměry obvodu
n_phys_qubits = loader.n_spin_orbitals
n_bond_qubits = 2
n_qubits = n_phys_qubits + n_bond_qubits

# 3. Sestavení vrstev jednoho QCMPS bloku: U -> CU -> U
u_layer = SingleQubitLayerType(UGate)
cu_layer = EntanglingLayerType(CUGate)
layers = [u_layer, cu_layer, u_layer]

hf = HFPreset(n_phys_qubits, occupied_indices=[0, 1, 2, 3])
qcmps = GlobalQCMPS(n_phys_qubits, n_bond_qubits, layers)

# 4. Sestavení celého ansatzu
ansatz = Ansatz(n_qubits)
ansatz.add_component(hf, list(range(n_bond_qubits, n_qubits)))
ansatz.add_component(qcmps, list(range(n_qubits)))

ansatz.add_hamiltonian(
    loader.hamiltonian,
    target_qubits=list(range(n_phys_qubits)),
)
ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)

print("Počáteční celková energie:", ansatz.total_energy())

# 5. Optimalizace parametrů QCMPS
result = minimize(
    ansatz.energy_objective(qcmps),
    qcmps.current_parameters(),
    method="COBYLA",
    options={"maxiter": 500},
)

qcmps.update_parameters(result.x)
print("Výsledná celková energie:", ansatz.total_energy())
```

Permutace Hamiltoniánu a obsazené orbitaly závisejí na konkrétním FCIDUMP souboru. Uvedené hodnoty platí pro přiložený `H4_FCIDUMP`.

## Hlavní části balíčku

- `Ansatz` skládá komponenty do jednoho kvantového obvodu a počítá energii nebo infidelitu vůči nulovému stavu.
- `GlobalQCMPS` vytváří globální QCMPS z počtu fyzických a bond qubitů a ze seznamu vrstev.
- `HFPreset` připravuje Hartreeův–Fockův výpočetní stav.
- `DoubleExcitation` přidává parametrizované dvojité excitace.
- `SingleQubitLayerType` aplikuje zvolený jednoqubitový typ brány na všechny qubity bloku.
- `EntanglingLayerType` aplikuje zvolený dvouqubitový typ brány na sousední qubity bloku.
- `FCIDumpLoader` načte FCIDUMP a převede fermionický Hamiltonián na qubitový operátor pomocí Jordanovy–Wignerovy transformace.

Vrstva nemusí používat pouze `UGate` a `CUGate`. V `qcmps.circuit_blocks` jsou také jednodušší jednoparametrové brány a brány založené na rotacích `X`, `Y` a `Z`.

Metoda `ansatz.energy_objective(component)` vytvoří účelovou funkci pouze pro vybranou komponentu. Ostatní komponenty zůstanou během této optimalizace fixované na svých aktuálních hodnotách. Po skončení je nutné uložit výsledek voláním `component.update_parameters(result.x)`.

## Experimenty

Připravené experimenty lze spustit z kořenového adresáře například takto:

```bash
python -m experiments.energy_optimization
python -m experiments.overlap_optimization
```

Souhrn dosavadních experimentů a jejich výsledků je v souboru [notes.md](notes.md).
