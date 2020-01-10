from pennylane import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize

def train(stepsize, steps, X, Y, var):
    opt = qml.AdamOptimizer(stepsize)
    params = []
    loss = []
    step = []
    for i in range(steps):
        var = opt.step(lambda v: cost(v, Xdata=X, Y=Y), var)
        params.append(var)
        loss.append(cost(var, Xdata=X, Y=Y))
        step.append(i+1)
    return print(step, params, loss)


def predict(Xnew, phi0, X, Y):
    newinput = Xnew
    x = np.delete(X, 0, 0)
    y = np.delete(Y, 0, 0)
    newinput = np.tile(newinput, (len(x), 1))  # create copies of new input
    Xdata = np.vstack((newinput, x))
    Ydata = np.tile(y, (2, 1))
    result1, result2 = circuit(phi0, Xdata, Ydata)
    return result2*result1


def test(Xtest, Ytest, Xdata, Y, var):
    count = 0
    x = np.delete(Xdata, 0, 0)
    y = np.delete(Y, 0, 0)
    Ydata = np.tile(y, (2, 1))
    for i in enumerate(Xtest):
        idx = i[0]
        newinput = i[1]
        newinput = np.tile(newinput, (len(x), 1))
        Xdata = np.vstack((newinput, x))
        result1, result2 = variational_circ(var, Xdata, Ydata)
        if np.round(result1*result2)==int(Ytest[idx]):
            count += 1
    accuracy = count/len(Xtest)
    return accuracy
