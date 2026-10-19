from qcmps.loader import FCIDumpLoader
from qcmps.ansatz import GlobalQCMPS, Ansatz, HFPreset
from qcmps.layers import *
from qcmps.gates import *
from qiskit.quantum_info import SparsePauliOp

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
    
    full_hamiltonian = SparsePauliOp.from_list([
        ("I" * num_bond_qubits + label, coefficient)
        for label, coefficient in loader.hamiltonian.to_list()
    ])

    ansatz = Ansatz(num_qubits)
    
    hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
    bw_layer = BrickWallLayerType(SO4Gate)
    qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, [bw_layer])
    
    print("Uniformly initializing QCMPS parameters...")
    qcmps.uniform_parameters()
    
    ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
    ansatz.add_component(qcmps, list(range(num_qubits)))
    
    ansatz.add_hamiltonian(full_hamiltonian, list(range(num_qubits)))
    ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)
    
    print("Initial total energy:")
    ansatz.print_total_energy()
    
    def print_optimization_energy(intermediate_result):
        total_energy = intermediate_result.fun + ansatz.nuclear_repulsion
        print(f"Total Energy: {total_energy:.12f}")  
    
    print("Starting QCMPS optimization...")
    minimization_result = minimize(
        fun = ansatz.energy_objective(qcmps), 
        x0 = qcmps.current_parameters(), 
        #jac = ansatz.energy_gradient_function(qcmps),
        method="L-BFGS-B", 
        options={"maxiter": 100000, "gtol": 1e-6}, 
        callback = print_optimization_energy
    )
    
    qcmps.update_parameters(minimization_result.x)
    
    print("Final total energy:")
    ansatz.print_total_energy()
    
    print("Adding additional layers to QCMPS...")
    qcmps.add_layers([bw_layer])
    ansatz.rebuild()
    
    print("Energy after adding layers:")
    ansatz.print_total_energy()
    
    print("Perturbing QCMPS parameters...")
    qcmps.perturb_parameters(0.001)
    
    print("Energy after perturbing parameters:")
    ansatz.print_total_energy()

    print("Starting QCMPS optimization after adding layers and perturbing parameters...")
    
    minimization_result = minimize(
        fun = ansatz.energy_objective(qcmps), 
        x0 = qcmps.current_parameters(), 
        #jac = ansatz.energy_gradient_function(qcmps),
        method="L-BFGS-B", 
        options={"maxiter": 100000, "gtol": 1e-6}, 
        callback = print_optimization_energy
    )
    
    qcmps.update_parameters(minimization_result.x)
    print("Final total energy after adding layers and perturbing parameters:")
    ansatz.print_total_energy()
    
if __name__ == "__main__":
    main()