from qcmps.loader import FCIDumpLoader
from qcmps.ansatz import GlobalQCMPS, Ansatz, HFPreset, DoubleExcitation, Component
from qcmps.circuit_blocks import LUBlock

from pathlib import Path
from scipy.optimize import minimize
import numpy as np

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    fcidump_path = PROJECT_ROOT / "fcidumps" / "H4_FCIDUMP"
    
    #Prepare the loader and permute the Hamiltonian to match the ordering of the qubits in the ansatz
    print("Loading the system")
    loader = FCIDumpLoader(fcidump_path)
    loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])
    
    hf_occupancy = [0, 1, 2, 3]
    
    num_phys_qubits = loader.n_spin_orbitals
    num_bond_qubits = 2
    num_qubits = num_phys_qubits + num_bond_qubits
    
    #Preparing the target double excitation initialization ansatz
    print("Preparing the double excitation initialization ansatz")
    
    de_ansatz = Ansatz(num_phys_qubits)
    
    hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    de = DoubleExcitation(num_phys_qubits, hf_occupancy)
    
    de_ansatz.add_component(hf_preset, list(range(num_phys_qubits)))
    de_ansatz.add_component(de, list(range(num_phys_qubits)))

    de_ansatz.add_hamiltonian(loader.hamiltonian, list(range(num_phys_qubits)))
    de_ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    def print_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + de_ansatz.nuclear_repulsion
        print(f"Total Energy: {total_energy:.12f}")
    
    de_ansatz.print_total_energy()
    print("Starting double excitation initialization optimization...")
    
    result = minimize(de_ansatz.energy_objective(de), de.current_parameters(), method="COBYLA", options={"maxiter": 90}, callback = print_optimization_energy)
    
    de.update_parameters(result.x)
    
    print(f"Double excitation initialization optimization completed with energy: {de_ansatz.total_energy():.12f}")
    
    #Save the optimized parameters for the double excitation initialization ansatz and the inverse of the target ansatz circuit to be used as a reference for the QCMPS ansatz
    optimized_parameters = de.current_parameters()
    de_inverse = de_ansatz.bound_circuit().inverse()
    
    #Preparing the overlap ansatz
    print("Preparing the overlap ansatz")
    overlap_ansatz = Ansatz(num_qubits)
    
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, block_type = LUBlock, layers = 2)
    de_inverse_fixed = Component.from_fixed_circuit(de_inverse)
    
    overlap_ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
    overlap_ansatz.add_component(de, list(range(num_bond_qubits, num_qubits)))
    overlap_ansatz.add_component(qcmps, list(range(num_qubits)))
    overlap_ansatz.add_component(de_inverse_fixed, list(range(num_phys_qubits)))
    
    #Start the step overlap optimization
    print("Starting step overlap optimization...")
    
    s = 0
    
    de.update_parameters(optimized_parameters*(1-s))
    
    print(f"Initial overlap should be 1.0: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
    
    #qcmps.uniform_parameters(min = 0.0, max = 0.1)
    
    callback_count = 0
    
    def print_optimization_overlap(intermediate_result):
        nonlocal callback_count
        callback_count += 1
        
        if callback_count % 10 == 0:
            overlap = 1 - intermediate_result.fun
            print(f"Evaluation {callback_count}: Overlap = {overlap:.12f}")
            
    steps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    for s in steps:
        
        print(f"Optimizing for s = {s:.3f}")
        
        de.update_parameters(optimized_parameters*(1-s))
        result = minimize(overlap_ansatz.zero_state_objective(qcmps, list(range(num_phys_qubits))), qcmps.current_parameters(), method="COBYLA", options={"maxiter": 1000}, callback = print_optimization_overlap)
        
        qcmps.update_parameters(result.x)
        
        print(f"Final overlap for s = {s:.2f}: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
        s += 0.05
        
    print(f"Final overlap: {1.0 - result.fun:.12f}")

if __name__ == "__main__":
    main()