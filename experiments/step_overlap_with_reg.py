from qcmps.loader import FCIDumpLoader
from qcmps.ansatz import GlobalQCMPS, Ansatz, HFPreset, DoubleExcitation, Component
from qcmps.circuit_blocks import UGate, CUGate, SingleQubitLayerType, EntanglingLayerType
from qcmps.logger import Logger

from pathlib import Path
from scipy.optimize import minimize
import numpy as np

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    fcidump_path = PROJECT_ROOT / "fcidumps" / "H4_FCIDUMP"
    
    logger = Logger("step_overlap_with_reg_log.txt")
    
    np.random.seed(42)  
    
    #Prepare the loader and permute the Hamiltonian to match the ordering of the qubits in the ansatz
    logger.log("Loading the system")
    loader = FCIDumpLoader(fcidump_path)
    loader.permute_hamiltonian([0, 2, 4, 6, 1, 3, 5, 7])
    
    hf_occupancy = [0, 1, 2, 3]
    
    num_phys_qubits = loader.n_spin_orbitals
    num_bond_qubits = 2
    num_qubits = num_phys_qubits + num_bond_qubits
    
    #Preparing the target double excitation initialization ansatz
    logger.log("Preparing the double excitation initialization ansatz")
    
    de_ansatz = Ansatz(num_phys_qubits)
    
    hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    de = DoubleExcitation(num_phys_qubits, hf_occupancy)
    
    de_ansatz.add_component(hf_preset, list(range(num_phys_qubits)))
    de_ansatz.add_component(de, list(range(num_phys_qubits)))

    de_ansatz.add_hamiltonian(loader.hamiltonian, list(range(num_phys_qubits)))
    de_ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    def log_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + de_ansatz.nuclear_repulsion
        logger.log(f"Total Energy: {total_energy:.12f}")
    
    logger.log(f"Total Energy before optimization: {de_ansatz.total_energy():.12f}")
    logger.log("Starting double excitation initialization optimization...")
    
    result = minimize(de_ansatz.energy_objective(de), de.current_parameters(), method="COBYLA", options={"maxiter": 90}, callback = log_optimization_energy)
    
    de.update_parameters(result.x)
    
    logger.log(f"Double excitation initialization optimization completed with energy: {de_ansatz.total_energy():.12f}")
    
    #Save the optimized parameters for the double excitation initialization ansatz and the inverse of the target ansatz circuit to be used as a reference for the QCMPS ansatz
    optimized_parameters = de.current_parameters()
    de_inverse = de_ansatz.bound_circuit().inverse()
    
    #Preparing the overlap ansatz
    logger.log("Preparing the overlap ansatz")
    overlap_ansatz = Ansatz(num_qubits)
    
    u_layer = SingleQubitLayerType(UGate)
    cu_layer = EntanglingLayerType(CUGate)
    
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, [u_layer, cu_layer, u_layer])
    de_inverse_fixed = Component.from_fixed_circuit(de_inverse)
    
    overlap_ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
    overlap_ansatz.add_component(de, list(range(num_bond_qubits, num_qubits)))
    overlap_ansatz.add_component(qcmps, list(range(num_qubits)))
    overlap_ansatz.add_component(de_inverse_fixed, list(range(num_phys_qubits)))
    
    #Start the step overlap optimization
    logger.log("Starting step overlap optimization...")
    
    s = 0
    
    de.update_parameters(optimized_parameters*(1-s))
    
    logger.log(f"Initial overlap should be 1.0: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
    
    qcmps.uniform_parameters(min = -0.001, max = 0.001)
    
    logger.log(f"Overlap after a small perturbation: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
        
    reg_epsilon = 0.5
    reg_lambda = 100
    reg_strength = 0.002
         
    def opt_function(x: np.ndarray) -> float:
        qcmps.update_parameters(x)
        
        infidelity = overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits)))
        
        sine = np.sin(x)
        
        regularization = 1/(reg_epsilon + reg_lambda * sine.dot(sine))
        
        return infidelity + reg_strength * regularization
    
    callback_count = 0
    
    def log_optimization_overlap(intermediate_result):
        nonlocal callback_count
        callback_count += 1
        
        if callback_count % 10 != 0:
            return
        
        x: np.ndarray = intermediate_result.x
        qcmps.update_parameters(x)
        
        infidelity = overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits)))
        overlap = 1 - infidelity
        regularization = 1/(reg_epsilon + reg_lambda * np.sin(x).dot(np.sin(x)))
        
        norm = np.linalg.norm(x)
        
        logger.log(f"Call {callback_count}: Overlap = {overlap:.12f}, Regularization = {regularization:.12f}, Objective = {intermediate_result.fun:.12f}, Norm = {norm:.12f}")
    
    hf_ansatz = Ansatz(num_phys_qubits)
    hf_ansatz.add_component(hf_preset, list(range(num_phys_qubits)))
    hf_ansatz.add_hamiltonian(loader.hamiltonian, list(range(num_phys_qubits)))
    hf_ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    qcmps_ansatz = Ansatz(num_qubits)
    qcmps_ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
    qcmps_ansatz.add_component(qcmps, list(range(num_qubits)))
    qcmps_ansatz.add_hamiltonian(loader.hamiltonian, list(range(num_phys_qubits)))
    qcmps_ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    steps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for s in steps:
        
        logger.log(f"Optimizing for s = {s:.3f} with regularization")
        
        de.update_parameters(optimized_parameters*(1-s))
        logger.log(f"Overlap before regularization for s = {s:.2f}: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
        logger.log(f"Energy of QCMPS ansatz before regularization for s = {s:.2f}: {qcmps_ansatz.total_energy():.12f}, Energy of HF ansatz: {hf_ansatz.total_energy():.12f}")
        result_regularized = minimize(opt_function, qcmps.current_parameters(), method="COBYLA", options={"maxiter": 500, "rhobeg": 0.01}, callback = log_optimization_overlap)
        
        qcmps.update_parameters(result_regularized.x)
        logger.log(f"Overlap after regularization for s = {s:.2f}: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
        logger.log(f"Energy of QCMPS ansatz after regularization for s = {s:.2f}: {qcmps_ansatz.total_energy():.12f}, Energy of HF ansatz: {hf_ansatz.total_energy():.12f}")
        
        logger.log(f"Optimizing for s = {s:.3f} without regularization")
        
        result_overlap = minimize(overlap_ansatz.zero_state_objective(qcmps, list(range(num_phys_qubits))), qcmps.current_parameters(), method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.001}, callback = log_optimization_overlap)
        
        qcmps.update_parameters(result_overlap.x)
        
        logger.log(f"Final overlap for s = {s:.2f}: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
        logger.log(f"Final energy of QCMPS ansatz for s = {s:.2f}: {qcmps_ansatz.total_energy():.12f}, Energy of HF ansatz: {hf_ansatz.total_energy():.12f}")
        
        reg_strength *= 0.5  # Reduce the regularization strength for the next step
        
    logger.log(f"Final overlap: {1 - overlap_ansatz.zero_state_infidelity(list(range(num_phys_qubits))):.12f}")
    logger.log(f"Final energy of QCMPS ansatz: {qcmps_ansatz.total_energy():.12f}, Energy of HF ansatz: {hf_ansatz.total_energy():.12f}")
    
    def log_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + qcmps_ansatz.nuclear_repulsion
        logger.log(f"Total Energy: {total_energy:.12f}")
        
    result_final_energy = minimize(qcmps_ansatz.energy_objective(qcmps), qcmps.current_parameters(), method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.001}, callback = log_optimization_energy)    

    qcmps.update_parameters(result_final_energy.x)
    
    logger.log(f"Final energy of QCMPS ansatz: {qcmps_ansatz.total_energy():.12f}, Energy of HF ansatz: {hf_ansatz.total_energy():.12f}")

if __name__ == "__main__":
    main()