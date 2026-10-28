

import numpy as np
import time
from typing import List, Callable, Tuple

class QuantumRegister:
    """
    Represents an n-qubit quantum register tracking 2^n complex probability amplitudes.
    """
    def __init__(self, num_qubits: int):
        if num_qubits < 1 or num_qubits > 30:
            raise ValueError("Qubit count must be between 1 and 30.")
        
        self.num_qubits = num_qubits
        self.dim = 1 << num_qubits
        # State vector initialized to |0...0>
        self.state_vector = np.zeros(self.dim, dtype=complex)
        self.state_vector[0] = 1.0 + 0.0j

    def normalize(self) -> None:
        """Normalizes the state vector to ensure total probability sums to 1."""
        norm = np.linalg.norm(self.state_vector)
        if norm == 0.0:
            raise ValueError("Zero-norm vector collapse detected.")
        self.state_vector /= norm

    def apply_single_qubit_gate(self, gate: np.ndarray, target_qubit: int) -> None:
        """
        Applies a single-qubit gate efficiently using tensor reshaping and broadcasting.
        """
        # Reshape state vector into axes for each qubit
        shape = [2] * self.num_qubits
        tensor_state = self.state_vector.reshape(shape)
        
        # Move target qubit to the front axis for matrix multiplication
        # Note: Python/NumPy indexing is 0-indexed from right-to-left or left-to-right depending on convention.
        # Here we map target_qubit to axis indexing (0 = most significant or least significant).
        axis = self.num_qubits - 1 - target_qubit
        
        # Contract gate matrix with the specific qubit axis: dot product along axis
        # tensor_state shape: [2, 2, 2, ...] -> dot product of gate (2x2) with axis
        tensor_state = np.tensordot(gate, tensor_state, axes=([1], [axis]))
        
        # Move the axis back to its original position
        tensor_state = np.moveaxis(tensor_state, 0, axis)
        
        # Flatten back into a 1D state vector
        self.state_vector = tensor_state.reshape(self.dim)

    def apply_cnot(self, control_qubit: int, target_qubit: int) -> None:
        """
        Applies a Controlled-NOT (CNOT) gate across the register.
        """
        ctrl_mask = 1 << control_qubit
        target_mask = 1 << target_qubit
        
        # Vectorized bit manipulation condition swap
        for i in range(self.dim):
            if (i & ctrl_mask) != 0:
                if (i & target_mask) == 0:
                    partner = i | target_mask
                    if i < partner:
                        # Swap amplitudes
                        temp = self.state_vector[i]
                        self.state_vector[i] = self.state_vector[partner]
                        self.state_vector[partner] = temp

    def measure(self) -> int:
        """
        Performs a quantum measurement, collapsing the wave-function into a basis state.
        """
        self.normalize()
        probabilities = np.abs(self.state_vector) ** 2
        collapsed_state = np.random.choice(self.dim, p=probabilities)
        
        # Collapse state vector
        self.state_vector = np.zeros(self.dim, dtype=complex)
        self.state_vector[collapsed_state] = 1.0 + 0.0j
        return collapsed_state


class QuantumGates:
    """Standard quantum mechanical gate matrices factory."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)

    @staticmethod
    .rz(theta: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0],
            [0.0, np.exp(1j * theta)]
        ], dtype=complex)


class QuantumCircuit:
    """Fluent API builder for chaining quantum operations."""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.operations: List[Callable[[QuantumRegister], None]] = []

    def h(self, qubit: int) -> 'QuantumCircuit':
        self.operations.append(lambda reg: reg.apply_single_qubit_gate(QuantumGates.H, qubit))
        return self

    def x(self, qubit: int) -> 'QuantumCircuit':
        self.operations.append(lambda reg: reg.apply_single_qubit_gate(QuantumGates.X, qubit))
        return self

    def z(self, qubit: int) -> 'QuantumCircuit':
        self.operations.append(lambda reg: reg.apply_single_qubit_gate(QuantumGates.Z, qubit))
        return self

    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        self.operations.append(lambda reg: reg.apply_single_qubit_gate(QuantumGates.rz(theta), qubit))
        return self

    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        self.operations.append(lambda reg: reg.apply_cnot(control, target))
        return self

    def execute(self) -> QuantumRegister:
        reg = QuantumRegister(self.num_qubits)
        for op in self.operations:
            op(reg)
        return reg


# ============================================================================
# EXECUTION HARNESS
# ============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("    QUANTUMCORE-PYTHON SIMULATOR ENGINE           ")
    print("==================================================")

    qubits = 4
    start_time = time.time()

    # Build an entangled multi-qubit circuit (GHZ State + Phase Rotation)
    circuit = QuantumCircuit(qubits)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    circuit.cnot(2, 3)
    circuit.rz(3, np.pi / 4.0)

    result_reg = circuit.execute()
    end_time = time.time()

    print(f"Executed {qubits}-qubit circuit successfully.")
    print(f"State Space Dimension: 2^{qubits} = {result_reg.dim} amplitudes.")
    print(f"Simulation Execution Time: {(end_time - start_time) * 1000:.3f} ms")

    measured_state = result_reg.measure()
    binary_representation = bin(measured_state)[2:].zfill(qubits)
    print(f"Post-Measurement Collapse State: |{binary_representation}> (Decimal: {measured_state})")
    print("==================================================")
