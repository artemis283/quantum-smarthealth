import pennylane as qml
import torch
import torch.nn as nn
from pytket.extensions.quantinuum import QuantinuumBackend

n_qubits = 3 # input size

dev = qml.device("default.qubit", wires=n_qubits)


# quantum node for the circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(input, weights):
    # encoding input features into quibit states
    for i in range(n_qubits):
        qml.RY(input[i], wires=i)

    # simple variational circuit with trainable rotations and entaglement
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # measure
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumModel(nn.Module):
    def __init__(self):
        super().__init__()
        #  initialising random weights for the circuit
        self.q_weights = nn.Parameter(torch.randn(n_qubits, n_qubits))
        self.fc = (nn.Linear(n_qubits, 1))

    def forward(self, x):
        q_out = torch.stack([
        torch.tensor(quantum_circuit(x[i], self.q_weights), dtype=torch.float32) for i in range(x.shape[0])])
        return torch.sigmoid(self.fc(q_out))
    
if __name__ == "__main__":
    model = QuantumModel()
    features = torch.tensor([0.1, 0.5, 0.2])  # example
    output = model(features.unsqueeze(0))    # shape [1, 3]
    print(output)
        

    
                                 




        
