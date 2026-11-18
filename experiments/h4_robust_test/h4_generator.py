from experiments._common import H4_FCIDUMP_PATH

from qcmps import (
    FCIDumpLoader,
    GlobalQCMPS, Ansatz, HFPreset,
    Logger,                      
    TwoSiteOptimizer,
    SO4Gate, BrickWallLayerType,
    Observable,
)

from pathlib import Path

def main():
    OUTPUT_DIR = Path(__file__).parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    runs = 10
    
    for i in range(runs):
        name = f"h4_{i}"
        checkpoint_path = OUTPUT_DIR / f"{name}_check.json"
        logging_path = OUTPUT_DIR / f"{name}.log"
        
        logger = Logger(logging_path, value_file=None, continue_logging=False)
        
        loader = FCIDumpLoader(H4_FCIDUMP_PATH)
        loader.interleave_spin_orbitals()

        hamiltonian = loader.hamiltonian
        nuclear_repulsion = loader.nuclear_repulsion
        hf_occupancy = loader.hf_occupancy()

        num_phys_qubits = loader.n_spin_orbitals
        num_bond_qubits = 1
        num_qubits = num_phys_qubits + num_bond_qubits

        physical_qubits = list(range(num_phys_qubits))
        
        ansatz = Ansatz(num_qubits)

        hf_preset = HFPreset(num_phys_qubits, hf_occupancy)
        bw_layer = BrickWallLayerType(SO4Gate)
        qcmps = GlobalQCMPS(num_phys_qubits, num_bond_qubits, [bw_layer])

        ansatz.add_component(hf_preset, list(range(num_bond_qubits, num_qubits)))
        ansatz.add_component(qcmps, list(range(num_qubits)))
        
        hamiltonian = Observable(hamiltonian, physical_qubits, constant = nuclear_repulsion, label = "Total energy")
        particle_number = Observable.particle_number(physical_qubits)
        alpha_number = Observable.alpha_particle_number(physical_qubits)
        beta_number = Observable.beta_particle_number(physical_qubits)
        
        energy = hamiltonian(ansatz)
        logger.log(f"HF Total energy: {energy:.12f}")
        
        qcmps.uniform_parameters()
        energy = hamiltonian(ansatz)
        logger.log(f"Initial energy: {energy:.12f}")
        
        logger.log("Starting Two site optimization...")
        global_optimizer = TwoSiteOptimizer(logger, log_every_n_steps = None, max_sweeps=50, local_maxiter=10, tolerance = 1e-6)
        result = global_optimizer.optimize(ansatz, hamiltonian, qcmps)
        
        logger.log(f"Total energy: {hamiltonian(ansatz):.12f}")
        logger.log(f"Total energy variance: {hamiltonian.var(ansatz):.12f}")
        logger.log(f"Particle number: {particle_number(ansatz):.12f}")
        logger.log(f"Particle number variance: {particle_number.var(ansatz):.12f}")
        logger.log(f"Alpha particle number: {alpha_number(ansatz):.12f}")
        logger.log(f"Beta particle number: {beta_number(ansatz):.12f}")
        
        ansatz.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()