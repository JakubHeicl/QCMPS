from .loader import FCIDumpLoader
from .ansatz import GlobalQCMPS, Ansatz, HFPreset, DoubleExcitationInit, Component
from .circuit_blocks import LUBlock

import numpy as np
from scipy.optimize import minimize

def test1():
    
    loader = FCIDumpLoader("fcidumps/H4_FCIDUMP")
    loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])
    
    hf_occupancy = [0, 1, 2, 3]
    
    num_phys_qubits = loader.n_spin_orbitals
    num_bond_qubits = 2
    num_qubits = num_phys_qubits + num_bond_qubits
    
    ansatz = Ansatz(num_qubits)
    
    hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    double_excitation_init = DoubleExcitationInit(num_phys_qubits, hf_occupancy)
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, block_type = LUBlock, layers = 1)
    
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
    
def test2():
    
    #Prepare the loader and permute the Hamiltonian to match the ordering of the qubits in the ansatz
    print("Loading the system")
    loader = FCIDumpLoader("fcidumps/H4_FCIDUMP")
    loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])
    
    hf_occupancy = [0, 1, 2, 3]
    
    num_phys_qubits = loader.n_spin_orbitals
    num_bond_qubits = 2
    num_qubits = num_phys_qubits + num_bond_qubits
    
    #Double excitation initialization
    print("Preparing the double excitation initialization ansatz")
    de_init_ansatz = Ansatz(num_phys_qubits)
    
    target_hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    target_de = DoubleExcitationInit(num_phys_qubits, hf_occupancy)
    
    de_init_ansatz.add_component(target_hf_preset, list(range(num_phys_qubits)))
    de_init_ansatz.add_component(target_de, list(range(num_phys_qubits)))
    
    de_init_ansatz.add_hamiltonian(loader.hamiltonian, list(range(num_phys_qubits)))
    de_init_ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    def print_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + de_init_ansatz.nuclear_repulsion
        print(f"Total Energy: {total_energy:.12f}")
    
    de_init_ansatz.print_total_energy()
    print("Starting double excitation initialization optimization...")
    
    result = minimize(de_init_ansatz.energy_objective(target_de), target_de.current_parameters(), method="COBYLA", options={"maxiter": 500}, callback = print_optimization_energy)
    
    target_de.update_parameters(result.x)
    
    #Inverse of the target ansatz circuit to be used as a reference for the QCMPS ansatz
    de_init_inverse = de_init_ansatz.bound_circuit().inverse()
    
    #Overlap ansatz
    print("Preparing the overlap ansatz")
    overlap_ansatz = Ansatz(num_qubits)
    
    overlap_hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, block_type = LUBlock, layers = 1)
    inverse_init = Component.from_fixed_circuit(de_init_inverse)
    
    overlap_ansatz.add_component(overlap_hf_preset, list(range(num_bond_qubits, num_qubits)))
    overlap_ansatz.add_component(qcmps, list(range(num_qubits)))
    overlap_ansatz.add_component(inverse_init, list(range(num_phys_qubits)))
    
    def print_optimization_overlap(intermediate_result):
        overlap = 1 - intermediate_result.fun
        print(f"Overlap: {overlap:.12f}")
    
    print("Starting QCMPS optimization for overlap...")
    result = minimize(overlap_ansatz.overlap_objective(qcmps, list(range(num_phys_qubits))), qcmps.current_parameters(), method="COBYLA", options={"maxiter": 500}, callback = print_optimization_overlap)
    
    qcmps.update_parameters(result.x)
    
    print(f"Final overlap: {1.0 - result.fun:.12f}")
    print(f"Evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
if __name__ == "__main__":
    test2()