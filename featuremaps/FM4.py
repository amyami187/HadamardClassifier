import pennylane as qml

def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[6, 5])  # flip label qubit
    # feature map
    qml.CRX(x1, wires=[6, 3])
    qml.CRX(phi0[0], wires=[6, 4])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 3])
    qml.CRY(x2, wires=[6, 3])
    qml.CRY(phi0[0], wires=[6, 4])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 3])
