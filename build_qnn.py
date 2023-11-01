import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.circuit import ParameterVector





# convolutional kernel 
def conv_circuit_0(params):
    target = QuantumCircuit(2)
    target.rx(params[0], 0)
    # target.ry(params[1], 0)
    # target.rz(params[1], 0)

    target.rx(params[1], 1)
    # target.ry(params[4], 1)
    # target.rz(params[3], 1)

    target.ryy(params[2], 0, 1)
    target.rzz(params[3], 0, 1)

    # target.rx(params[7], 0)
    # target.ry(params[5], 0)
    target.rz(params[4], 0)

    # target.rx(params[10], 1)
    # target.ry(params[7], 1)
    target.rz(params[5], 1)


    # target.rz(-np.pi / 2, 1)
    # target.cx(1, 0)
    # target.rx(params[0], 0)
    # target.ry(params[1], 1)
    # target.cx(0, 1)
    # target.ry(params[2], 1)
    # target.cx(1, 0)
    # target.rz(np.pi / 2, 0)

    # target.rx(params[3], 0)
    # target.rx(params[4], 1)
    return target


def conv_circuit(params):
    target = QuantumCircuit(2)
    target.x(0)
    target.sx(0)
    target.rx(params[0], 0)
    # target.ry(params[1], 0)
   

    target.x(1)
    target.sx(1)
    target.rx(params[1], 1)
    # target.ry(params[4], 1)


    target.ryy(params[2], 0, 1)
    target.rzz(params[3], 0, 1)
    target.cx(0, 1)

    # target.rx(params[7], 0)
    # target.ry(params[5], 0)
    target.x(0)
    target.sx(0)
    target.rz(params[4], 0)

    # target.rx(params[10], 1)
    # target.ry(params[7], 1)
    target.x(1)
    target.sx(1)
    target.rz(params[5], 1)

    target.cx(1, 0)


    # target.rz(-np.pi / 2, 1)
    # target.cx(1, 0)
    # target.rx(params[0], 0)
    # target.ry(params[1], 1)
    # target.cx(0, 1)
    # target.ry(params[2], 1)
    # target.cx(1, 0)
    # target.rz(np.pi / 2, 0)

    # target.rx(params[3], 0)
    # target.rx(params[4], 1)
    return target


# convolutional layer
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    kernel_para_num = 6
    params = ParameterVector(param_prefix, length=num_qubits * kernel_para_num)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + kernel_para_num)]), [q1, q2])
        qc.barrier()
        param_index += kernel_para_num
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + kernel_para_num)]), [q1, q2])
        qc.barrier()
        param_index += kernel_para_num

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params):
    target = QuantumCircuit(2)
    # target.rz(-np.pi / 2, 1)
    # target.cx(1, 0)
    # target.rz(params[0], 0)
    # target.ry(params[1], 1)
    target.cx(0, 1)
    target.x(1)
    target.sx(1)
    target.ry(params[0], 1)

    return target


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 1

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc






def z_expectation_value(output_probs):

    return output_probs


def build_qcnn_4q(quantum_instance):
    num_qubits = 4
    feature_map = ZFeatureMap(num_qubits)
    # feature_map.decompose().draw("mpl")

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(4, "с1"), list(range(4)), inplace=True)  
    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p1"), list(range(4)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(2, "c2"), list(range(2, 4)), inplace=True)

    ansatz.compose(pool_layer([0], [1], "p2"), list(range(2, 4)), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    c = ClassicalRegister(1, name='c')
    circuit.add_register(c)

    circuit.measure(3, 0)

    qnn = CircuitQNN(
        circuit=circuit.decompose(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=z_expectation_value,
        output_shape=(2,),
        quantum_instance=quantum_instance
    )

    return qnn

