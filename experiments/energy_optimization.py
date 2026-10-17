from qcmps.loader import FCIDumpLoader
from qcmps.ansatz import GlobalQCMPS, Ansatz, HFPreset, DoubleExcitation
from qcmps.circuit_blocks import UGate, CUGate, SingleQubitLayerType, EntanglingLayerType

import numpy as np
from scipy.optimize import minimize
from pathlib import Path

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    fcidump_path = PROJECT_ROOT / "fcidumps" / "H4_FCIDUMP"
    
    loader = FCIDumpLoader(fcidump_path)
    loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])

    hf_occupancy = [0, 1, 2, 3]

    num_phys_qubits = loader.n_spin_orbitals
    num_bond_qubits = 2
    num_qubits = num_phys_qubits + num_bond_qubits

    ansatz = Ansatz(num_qubits)

    hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    double_excitation_init = DoubleExcitation(num_phys_qubits, hf_occupancy)
    
    u_layer = SingleQubitLayerType(UGate)
    cu_layer = EntanglingLayerType(CUGate)
    
    layers = [u_layer, cu_layer, u_layer]
    
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, layers)

    ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
    ansatz.add_component(double_excitation_init, list(range(num_bond_qubits, num_qubits)))
    ansatz.add_component(qcmps, list(range(num_qubits)))

    ansatz.add_hamiltonian(loader.hamiltonian, list(range(num_phys_qubits)))
    ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)

    en_before = ansatz.total_energy()
    ansatz.print_total_energy()

    def print_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + ansatz.nuclear_repulsion
        print(f"Total Energy: {total_energy:.12f}")

    print("Starting init optimization...")

    minimization_result = minimize(ansatz.energy_objective(double_excitation_init), double_excitation_init.current_parameters(), method="COBYLA", options={"maxiter": 90}, callback = print_optimization_energy)

    print("Init optimization finished.")

    double_excitation_init.update_parameters(minimization_result.x)
    en_after_init = ansatz.total_energy()
    ansatz.print_total_energy()

    print("Starting QCMPS optimization...")

    minimization_result = minimize(ansatz.energy_objective(qcmps), qcmps.current_parameters(), method="COBYLA", options={"maxiter": 500}, callback = print_optimization_energy)
        
    qcmps.update_parameters(minimization_result.x)
    en_after_qcmps = ansatz.total_energy()
    ansatz.print_total_energy()

    print("QCMPS optimization finished.")

    print(f"Energy before optimization: {en_before:.12f}")
    print(f"Energy after init optimization: {en_after_init:.12f}")
    print(f"Energy after QCMPS optimization: {en_after_qcmps:.12f}")

if __name__ == "__main__":
    main()