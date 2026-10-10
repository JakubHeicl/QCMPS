from .loader import FCIDumpLoader
from .qcmps_ansatz import GlobalAnsatz
from .circuit_blocks import LUBlock

import numpy as np
from scipy.optimize import minimize

def main():
    
    loader = FCIDumpLoader("fcidumps/H4_FCIDUMP")
    loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])
    
    hf_occupancy = [0, 1, 2, 3]
    
    ansatz = GlobalAnsatz(
        n_orbs=loader.n_spin_orbitals,
        n_bond_qubits=2,
        block_type=LUBlock,
        layers=1,
        hf_occ=hf_occupancy,
        double_excitation=True,
    )
    
    ansatz.add_hamiltonian(loader.hamiltonian)
    ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    ansatz.print_total_energy()
    en_before = ansatz.total_energy()
    
    print("Starting init optimization...")

    minimization_result = minimize(ansatz.init_opt_function(), ansatz.current_init_parameters(), method="COBYLA", options={"maxiter": 90}, callback = ansatz.print_total_energy)
    
    print("Init optimization finished.")
    
    ansatz.update_init_parameters(minimization_result.x)
    en_after_init = ansatz.total_energy()
    ansatz.print_total_energy()
    
    print("Starting QCMPS optimization...")
    
    minimization_result = minimize(ansatz.qcmps_opt_function(), ansatz.current_qcmps_parameters(), method="COBYLA", options={"maxiter": 500}, callback = ansatz.print_total_energy)
    
    ansatz.update_qcmps_parameters(minimization_result.x)
    en_after_qcmps = ansatz.total_energy()
    ansatz.print_total_energy()
    
    print("QCMPS optimization finished.")
    
    print(f"Energy before optimization: {en_before:.12f}")
    print(f"Energy after init optimization: {en_after_init:.12f}")
    print(f"Energy after QCMPS optimization: {en_after_qcmps:.12f}")
    
if __name__ == "__main__":
    main()