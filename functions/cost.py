def cost(var, X, Y):
    MSE = 0
    for i in enumerate(X):
        idx = i[0]
        newinput = i[1]
        x = np.delete(X, i[0], 0)  #  remove chosen data point (which is the "new input") so that only the other 4 data points go into the cost
        if idx == len(x) - 1: break
        newinput = np.tile(newinput, (len(x), 1))  # create copies of new input
        Xdata = np.vstack((newinput, x))  # stack 4 data points and copies of new input for the circuit
        ytrue = int(Y[idx])  # select the true y label of the new input
        y = np.delete(Y, idx, 0)  # remove chosen data point's label
        Ydata = np.tile(y, (2, 1)) # Create copy of labels
        result1, result2 = circuit(var,Xdata,Ydata)
        ypred = result1 * result2
        loss = (ypred - ytrue) ** 2
        MSE = MSE + loss
    print(MSE)
    return MSE / len(x)
