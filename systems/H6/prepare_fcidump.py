from __future__ import annotations

import json
from math import isclose
from pathlib import Path

from pyscf import gto, mcscf, scf, tools


OUTPUT_DIR = Path(__file__).resolve().parent
FCIDUMP_PATH = OUTPUT_DIR / "H6_FCIDUMP"
REFERENCE_PATH = OUTPUT_DIR / "H6_reference.json"
CHECKPOINT_PATH = OUTPUT_DIR / "H6_reference.chk"

# The nuclear-repulsion energy stored in H6_FCIDUMP_OLD corresponds to a
# linear H6 chain with a uniform nearest-neighbour distance of 2 Angstrom.
ATOM = "\n".join(f"H 0 0 {2 * index}" for index in range(6))
BASIS = "sto-3g"
CHARGE = 0
SPIN = 0
UNIT = "Angstrom"

EXPECTED_NUCLEAR_REPULSION = 2.3019208561050002


def main() -> None:
    molecule = gto.M(
        atom=ATOM,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        symmetry=False,
        verbose=4,
    )

    nuclear_repulsion = float(molecule.energy_nuc())
    if not isclose(
        nuclear_repulsion,
        EXPECTED_NUCLEAR_REPULSION,
        rel_tol=0.0,
        abs_tol=2e-8,
    ):
        raise RuntimeError(
            "The reconstructed H6 geometry does not match the nuclear "
            "repulsion stored in H6_FCIDUMP_OLD."
        )

    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.chkfile = str(CHECKPOINT_PATH)
    mean_field.kernel()

    if not mean_field.converged:
        raise RuntimeError("The H6 RHF calculation did not converge.")

    ncas = mean_field.mo_coeff.shape[1]
    nelecas = molecule.nelectron

    casci = mcscf.CASCI(mean_field, ncas, nelecas)
    if hasattr(casci.fcisolver, "conv_tol"):
        casci.fcisolver.conv_tol = 1e-12
    casci.kernel()

    tools.fcidump.from_mcscf(casci, str(FCIDUMP_PATH))
    fcidump = tools.fcidump.read(str(FCIDUMP_PATH), verbose=False)

    if int(fcidump["NORB"]) != 6 or int(fcidump["NELEC"]) != 6:
        raise RuntimeError(
            "Generated an unexpected active space: expected NORB=6 and "
            "NELEC=6."
        )

    metadata = {
        "atom": ATOM,
        "basis": BASIS,
        "charge": CHARGE,
        "spin_2S": SPIN,
        "unit": UNIT,
        "scf_method": mean_field.__class__.__name__,
        "correlated_method": "CASCI",
        "ncas": int(ncas),
        "nelecas": int(nelecas),
        "scf_total_energy_hartree": float(mean_field.e_tot),
        "reference_total_energy_hartree": float(casci.e_tot),
        "fcidump_norb": int(fcidump["NORB"]),
        "fcidump_nelec": int(fcidump["NELEC"]),
        "fcidump_ecore_hartree": float(fcidump["ECORE"]),
        "old_fcidump_ecore_hartree": EXPECTED_NUCLEAR_REPULSION,
    }

    REFERENCE_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {FCIDUMP_PATH}")
    print(f"Wrote {REFERENCE_PATH}")
    print(f"RHF total energy:       {mean_field.e_tot:.16f} Eh")
    print(f"CASCI total energy:     {casci.e_tot:.16f} Eh")
    print(f"FCIDUMP ECORE:          {fcidump['ECORE']:.16f} Eh")


if __name__ == "__main__":
    main()
