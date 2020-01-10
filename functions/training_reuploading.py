from pennylane import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize

def variational_circ(var, Xdata, Y):
    phi0 = var[0]
    x_weights = var[1]
    Xdata = x_weights * Xdata
    return circuit(phi0, Xdata, Y)


def cost(var, Xdata, Y):
    MSE = 0
    for i in enumerate(X):
        idx = i[0]
        newinput = i[1]
        x = np.delete(Xdata, i[0], 0)  #  remove chosen data point (which is the "new input") so that only the other 4 data points go into the cost
        if idx == len(x) - 1: break
        newinput = np.tile(newinput, (len(x), 1))  # create copies of new input
        Xdata = np.vstack((newinput, x))  # stack 4 data points and copies of new input for the circuit
        ytrue = int(Y[idx])  # select the true y label of the new input
        y = np.delete(Y, idx, 0)  # remove chosen data point's label
        Ydata = np.tile(y, (2, 1)) # Create copy of labels
        result1, result2 = variational_circ(var=var, Xdata=Xdata, Y=Ydata)
        ypred = result1 * result2
        loss = (ypred - ytrue) ** 2
        MSE = MSE + loss
    print(MSE)
    return MSE / len(x)


def var_init():
    var1 = 0.01 * np.random.randn(12, 1)
    var2 = 0.01 * np.random.randn(8, 2)
    var = [var1, var2]
    return var


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
