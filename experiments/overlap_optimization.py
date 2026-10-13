from qcmps.loader import FCIDumpLoader
from qcmps.ansatz import GlobalQCMPS, Ansatz, HFPreset, DoubleExcitationInit, Component
from qcmps.circuit_blocks import LUBlock

from pathlib import Path
from scipy.optimize import minimize

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
    
    result = minimize(de_init_ansatz.energy_objective(target_de), target_de.current_parameters(), method="COBYLA", options={"maxiter": 5000}, callback = print_optimization_energy)
    
    target_de.update_parameters(result.x)
    
    #Inverse of the target ansatz circuit to be used as a reference for the QCMPS ansatz
    de_init_inverse = de_init_ansatz.bound_circuit().inverse()
    
    #Overlap ansatz
    print("Preparing the overlap ansatz")
    overlap_ansatz = Ansatz(num_qubits)
    
    overlap_hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, block_type = LUBlock, layers = 1)
    qcmps.uniform_parameters()
    inverse_init = Component.from_fixed_circuit(de_init_inverse)
    
    overlap_ansatz.add_component(overlap_hf_preset, list(range(num_bond_qubits, num_qubits)))
    overlap_ansatz.add_component(qcmps, list(range(num_qubits)))
    overlap_ansatz.add_component(inverse_init, list(range(num_phys_qubits)))
    
    def print_optimization_overlap(intermediate_result):
        overlap = 1 - intermediate_result.fun
        print(f"Overlap: {overlap:.12f}")
    
    print("Starting QCMPS optimization for overlap...")
    result = minimize(overlap_ansatz.zero_state_objective(qcmps, list(range(num_phys_qubits))), qcmps.current_parameters(), method="COBYLA", options={"maxiter": 5000}, callback = print_optimization_overlap)
    
    qcmps.update_parameters(result.x)
    
    print(f"Final overlap: {1.0 - result.fun:.12f}")
    print(f"Evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
if __name__ == "__main__":
    main()