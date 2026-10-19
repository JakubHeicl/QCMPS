from qcmps.loader import FCIDumpLoader
from qcmps.ansatz import GlobalQCMPS, Ansatz, HFPreset
from qcmps.layers import *
from qcmps.gates import *
from qcmps.logger import Logger

from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from experiments._common import H4_FCIDUMP_PATH, H6_FCIDUMP_PATH

def main():
    loader = FCIDumpLoader(H6_FCIDUMP_PATH)
    loader.interleave_spin_orbitals()
    logger = Logger("energy_so4_h6.log")
    
    hf_occupancy = loader.hf_occupancy()

    num_phys_qubits = loader.n_spin_orbitals
    num_bond_qubits = 2
    num_qubits = num_phys_qubits + num_bond_qubits
    
    full_hamiltonian = SparsePauliOp.from_list([
        ("I" * num_bond_qubits + label, coefficient)
        for label, coefficient in loader.hamiltonian.to_list()
    ])

    ansatz = Ansatz(num_qubits)
    
    hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    bw_layer = BrickWallLayerType(SO4Gate)
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, [bw_layer])
    
    ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
    ansatz.add_component(qcmps, list(range(num_qubits)))
    
    ansatz.add_hamiltonian(full_hamiltonian, list(range(num_qubits)))
    ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    logger.log("HF energy:")
    logger.log(f"{ansatz.total_energy():.12f}")
    
    logger.log("Initial total energy after uniform initialization:")
    qcmps.uniform_parameters()
    logger.log(f"{ansatz.total_energy():.12f}")
    
    def print_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + ansatz.nuclear_repulsion
        logger.log(f"[opt] Total Energy: {total_energy:.12f}")  
    
    logger.log("Starting QCMPS optimization...")
    minimization_result = minimize(
        fun = ansatz.energy_objective(qcmps), 
        x0 = qcmps.current_parameters(), 
        #jac = ansatz.energy_gradient_function(qcmps),
        method="L-BFGS-B", 
        options={"maxiter": 100000, "gtol": 1e-6}, 
        callback = print_optimization_energy
    )
    
    qcmps.update_parameters(minimization_result.x)
    
    logger.log("Final total energy:")
    logger.log(f"{ansatz.total_energy():.12f}")
    
    logger.log("Adding additional layers to QCMPS...")
    qcmps.add_layers([bw_layer])
    ansatz.rebuild()
    
    print("Energy after adding layers:")
    logger.log(f"{ansatz.total_energy():.12f}")
    
    logger.log("Perturbing QCMPS parameters...")
    qcmps.perturb_parameters(0.001)
    
    logger.log("Energy after perturbing parameters:")
    logger.log(f"{ansatz.total_energy():.12f}")

    logger.log("Starting QCMPS optimization after adding layers and perturbing parameters...")
    
    minimization_result = minimize(
        fun = ansatz.energy_objective(qcmps), 
        x0 = qcmps.current_parameters(), 
        #jac = ansatz.energy_gradient_function(qcmps),
        method="L-BFGS-B", 
        options={"maxiter": 100000, "gtol": 1e-6}, 
        callback = print_optimization_energy
    )
    
    qcmps.update_parameters(minimization_result.x)
    logger.log("Final total energy after adding layers and perturbing parameters:")
    logger.log(f"{ansatz.total_energy():.12f}")

if __name__ == "__main__":
    main()
