from __future__ import annotations
from qiskit import QuantumCircuit

class HFPreSet(QuantumCircuit):
    def __init__(self, n_entries: int):

        super().__init__(n_entries)
        self.x(0)

class Gate(QuantumCircuit):
    def __init__(self, parameters: list[float], n_params: int, n_qubits: int):
        super().__init__(n_qubits)

        if len(parameters) != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}.")
    
    @staticmethod
    def n_parameters() -> int:
        raise NotImplementedError("Subclasses must implement n_parameters()")

class UGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 3, 2)

        a, b, c = parameters

        self.cx(0, 1)
        self.ry(a, 0)
        self.rz(b, 0)
        self.ry(c, 0)
        self.cx(0, 1)

    @staticmethod
    def n_parameters() -> int:
        return 3

class Block(QuantumCircuit):

    gate_cls: type[Gate] | None = None

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries)

        if self.gate_cls is None:
            raise ValueError("gate_cls must be set before instantiating Block subclasses.")
       
        self.layers = layers
        self.gate_parameters = self.gate_cls.n_parameters()

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        raise NotImplementedError("Subclasses must implement n_parameters()")

class AUBlock(Block):

    gate_cls = UGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries):
                for j in range(n_entries):

                    if i == j:
                        continue

                    gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                    index += self.gate_parameters

                    self.append(gate_circuit.to_gate(), [i, j])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * n_entries * (n_entries - 1) * layers

class LUBlock(Block):
    
    gate_cls = UGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries - 1):

                gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                index += self.gate_parameters

                self.append(gate_circuit.to_gate(), [i, i + 1])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * (n_entries - 1) * layers