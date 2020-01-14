from pennylane import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize
from circuits.qcirc_4datapoints import *

def train(stepsize, steps, Xdata, Ydata, var):
    opt = qml.AdamOptimizer(stepsize)
    params = []
    loss = []
    step = []
    for i in range(steps):
        var = opt.step(lambda v: cost(v, X=Xdata, Y=Ydata), var)
        params.append(var)
        print(var)
        loss.append(cost(var, X=Xdata, Y=Ydata))
        step.append(i+1)
    return print(step, params, loss)


def predict(Xnew, var, X, Y):
    newinput = Xnew
    x = np.delete(X, 0, 0)
    y = np.delete(Y, 0, 0)
    newinput = np.tile(newinput, (len(x), 1))  # create copies of new input
    Xdata = np.vstack((newinput, x))
    Ydata = np.tile(y, (2, 1))
    result1, result2 = circuit(var, Xdata, Ydata)
    return result2*result1


def test(Xtest, Ytest, X, Y, var):
    count = 0
    x = np.delete(X, 0, 0)
    y = np.delete(Y, 0, 0)
    Ydata = np.tile(y, (2, 1))
    for i in enumerate(Xtest):
        idx = i[0]
        newinput = i[1]
        newinput = np.tile(newinput, (len(x), 1))
        Xdata = np.vstack((newinput, x))
        result1, result2 = circuit(var, X=Xdata, Y=Ydata)
        if np.round(result1*result2)==int(Ytest[idx]):
            count += 1
    accuracy = count/len(Xtest)
    return accuracy

def cost(var, X, Y): # specifically for dataset = 5 points # 4 data points and 1 test input
    MSE = 0
    for i in enumerate(X):
        idx = i[0]
        newinput = i[1]
        x = np.delete(X, i[0], 0)  #  remove chosen data point (which is the "new input")
        if idx == len(x) - 1: break
        newinput = np.tile(newinput, (len(x), 1))  # create copies of new input
        Xdatacost = np.vstack((newinput, x))  # stack 4 data points and copies of new input for the circuit
        ytrue = int(Y[idx])  # select the true y label of the new input
        y = np.delete(Y, idx, 0)  # remove chosen data point's label
        Ydatacost = np.tile(y, (2, 1)) # Create copy of labels
        result1, result2 = circuit(var, X=Xdatacost, Y=Ydatacost)
        ypred = result1 * result2
        loss = (ypred - ytrue) ** 2
        MSE = MSE + loss
    return MSE / len(x)
