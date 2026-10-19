from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit.quantum_info import SparsePauliOp

from pathlib import Path

class FCIDumpLoader:
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        
        self.problem = fcidump_to_problem(FCIDump.from_file(filepath))
        
        hamiltonian = self.problem.second_q_ops()[0]
        mapper = JordanWignerMapper()
        self.hamiltonian = mapper.map(hamiltonian).simplify()
        
        self.nuclear_repulsion = self.problem.nuclear_repulsion_energy
            
        self.n_spin_orbitals = self.problem.num_spin_orbitals
        self.n_spatial_orbitals = self.n_spin_orbitals // 2
    
        self.num_alpha = int(self.problem.num_alpha)
        self.num_beta = int(self.problem.num_beta)
        self.num_particles_total = self.num_alpha + self.num_beta
        
    def permute_hamiltonian(self, permutation: list[int]):
        
        n_qubits = self.hamiltonian.num_qubits
        
        permuted_terms = []
        
        for label, coeff in self.hamiltonian.to_list():
            permuted = ["I"] * n_qubits
            for old_qubit, new_qubit in enumerate(permutation):
                permuted[n_qubits - 1 - new_qubit] = label[n_qubits - 1 - old_qubit]
            permuted_terms.append(("".join(permuted), coeff))
    
        self.hamiltonian = SparsePauliOp.from_list(permuted_terms).simplify()

    def interleave_spin_orbitals(self) -> None:
        n = self.n_spatial_orbitals
        
        permutation = (
            [2 * orbital for orbital in range(n)] +
            [2 * orbital + 1 for orbital in range(n)]
        )
        
        self.permute_hamiltonian(permutation)
    
    def hf_occupancy(self) -> list[int]:
        return (
            [2 * orbital for orbital in range(self.num_alpha)] + 
            [2 * orbital + 1 for orbital in range(self.num_beta)]
        )
        