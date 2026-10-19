from pathlib import Path
import json

from pyscf import gto, scf, mcscf, tools

# =========================
# USER SETTINGS
# =========================
atom = """
H 0 0 0
H 0 0 0.7414
H 0 0 2.0643
H 0 0 2.8057
"""
basis = "sto-3g"
charge = 0
spin = 0          # 2S
unit = "Angstrom"

# None => full-space FCI in HF MO basis (tiny systems only)
ncas = None
nelecas = None

# For reproducible tests prefer CASCI=False? No:
# use_casscf=False means CASCI (fixed orbitals, better for regression tests)
use_casscf = False

fcidump_name = "H4_FCIDUMP"
meta_name = "reference.json"

# =========================
# BUILD MOLECULE + SCF
# =========================
mol = gto.M(
    atom=atom,
    basis=basis,
    charge=charge,
    spin=spin,
    unit=unit,
    symmetry=False,
    verbose=4,
)

mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.chkfile = "reference.chk"
mf.kernel()

if not mf.converged:
    raise RuntimeError("SCF did not converge.")

# =========================
# DEFINE ACTIVE SPACE
# =========================
norb = mf.mo_coeff.shape[1]

if ncas is None:
    ncas = norb
    nelecas = mol.nelectron

if nelecas is None:
    raise ValueError("When ncas is set explicitly, nelecas must also be set.")

# =========================
# CASCI / CASSCF
# =========================
if use_casscf:
    mc = mcscf.CASSCF(mf, ncas, nelecas)
else:
    mc = mcscf.CASCI(mf, ncas, nelecas)

if hasattr(mc.fcisolver, "conv_tol"):
    mc.fcisolver.conv_tol = 1e-12

mc.kernel()

# =========================
# WRITE FCIDUMP
# =========================
tools.fcidump.from_mcscf(mc, fcidump_name)

# Read it back for metadata / sanity check
fc = tools.fcidump.read(fcidump_name, verbose=False)

meta = {
    "atom": atom.strip(),
    "basis": basis,
    "charge": charge,
    "spin_2S": spin,
    "unit": unit,
    "scf_method": mf.__class__.__name__,
    "correlated_method": "CASSCF" if use_casscf else "CASCI",
    "ncas": int(ncas),
    "nelecas": nelecas,
    "scf_total_energy_hartree": float(mf.e_tot),
    "reference_total_energy_hartree": float(mc.e_tot),
    "fcidump_norb": int(fc["NORB"]),
    "fcidump_nelec": int(fc["NELEC"]),
    "fcidump_ecore_hartree": float(fc["ECORE"]),
}

Path(meta_name).write_text(json.dumps(meta, indent=2))

print(f"Wrote {fcidump_name}")
print(f"Wrote {meta_name}")
print(f"Reference total energy: {mc.e_tot:.16f} Eh")
print(f"FCIDUMP ECORE:         {fc['ECORE']:.16f} Eh")