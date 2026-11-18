from pathlib import Path

from qcmps import (
    FCIDumpLoader,
    GlobalQCMPS, Ansatz, HFPreset,
    SO4Gate, BrickWallLayerType,
    Observable,
    Component
)

from experiments._common import H4_FCIDUMP_PATH

def main():
    OUTPUT_DIR = Path(__file__).parent / "output"
    
    runs = 6 
    
    lowest_overlap = 1.0
    
    for i in range(runs):
        for j in range(i+1, runs):
            
            name_i = f"h4_{i}"
            name_j = f"h4_{j}"
            
            loader = FCIDumpLoader(H4_FCIDUMP_PATH)
            loader.interleave_spin_orbitals()

            hf_occupancy = loader.hf_occupancy()

            num_phys_qubits = loader.n_spin_orbitals
            num_bond_qubits = 1
            num_qubits = num_phys_qubits + num_bond_qubits

            physical_qubits = list(range(num_phys_qubits))
            
            ansatz_i = Ansatz(num_qubits)
            ansatz_j = Ansatz(num_qubits)

            hf_preset_i = HFPreset(num_phys_qubits, hf_occupancy)
            hf_preset_j = HFPreset(num_phys_qubits, hf_occupancy)
            
            bw_layer = BrickWallLayerType(SO4Gate)
            qcmps_i = GlobalQCMPS(num_phys_qubits, num_bond_qubits, [bw_layer])
            qcmps_j = GlobalQCMPS(num_phys_qubits, num_bond_qubits, [bw_layer])

            ansatz_i.add_component(hf_preset_i, list(range(num_bond_qubits, num_qubits)))
            ansatz_i.add_component(qcmps_i, list(range(num_qubits)))

            ansatz_j.add_component(hf_preset_j, list(range(num_bond_qubits, num_qubits)))
            ansatz_j.add_component(qcmps_j, list(range(num_qubits)))
            
            ansatz_i.load_checkpoint(OUTPUT_DIR / f"{name_i}_check.json")
            ansatz_j.load_checkpoint(OUTPUT_DIR / f"{name_j}_check.json")
            
            zero_state_infidelity = Observable.zero_state_infidelity(list(range(num_qubits)))
            
            ansatz_j_inv = ansatz_j.bound_circuit().inverse()
            
            ansatz_j_inv = Component.from_fixed_circuit(ansatz_j_inv)
            
            ansatz_i.add_component(ansatz_j_inv, list(range(num_qubits)))
            
            overlap = 1 - zero_state_infidelity(ansatz_i)
            
            print(f"Overlap between {name_i} and {name_j}: {overlap:.12f}")
            
            if overlap < lowest_overlap:
                lowest_overlap = overlap

    print(f"Lowest overlap: {lowest_overlap:.12f}")

if __name__ == "__main__":
    main()