import preprocesso as pre
import numpy as np
import pandas as pd

#temporizador
import time
from functools import wraps

def computeSupportVectorRegressionModel(X, y, k, d):
    from sklearn.svm import SVR
    if k == "poly":
        regressor = SVR(kernel = k, degree=d)
    else:
        regressor = SVR(kernel = k)
    regressor.fit(X,np.ravel(y))
    return regressor

def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color = 'red') #plot real y points
    plt.plot(XLine, yLine, color = 'blue') #plot predicted points in line
    plt.title("Comparando pontos reais com a reta produzida pela regressão vetor suporte")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()


def runSupportVectorRegressionExample(filename):
    start_time = time.time()
    X, y = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.fillMissingData(X, 0, 1)
    elapsed_time = time.time() - start_time
    print("Fill Missing Data: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X, scaleX = pre.computeScaling(X)
    y, scaleY = pre.computeScaling(np.reshape(y,(-1,1)))
    elapsed_time = time.time() - start_time
    print("Compute Scaling: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "linear", 2)
    elapsed_time = time.time() - start_time
    print("Compute Support Vector Regression with kernel Linear: %.2f" % elapsed_time, "segundos.")
    
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), scaleX.inverse_transform(X), scaleY.inverse_transform(np.reshape((svrModel.predict(X)),(-1,1))))

    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "poly", 3)
    elapsed_time = time.time() - start_time
    print("Compute Support Vector Regression with kernel Poly: %.2f" % elapsed_time, "segundos.")
    
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), scaleX.inverse_transform(X), scaleY.inverse_transform(np.reshape((svrModel.predict(X)),(-1,1))))

    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "rbf", 2)
    elapsed_time = time.time() - start_time
    print("Compute Support Vector Regression with kernel RBF: %.2f" % elapsed_time, "segundos.")
    
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), scaleX.inverse_transform(X), scaleY.inverse_transform(np.reshape((svrModel.predict(X)),(-1,1))))

    from sklearn.metrics import r2_score
    return r2_score(y, svrModel.predict(X))

if __name__ == "__main__":
    print(runSupportVectorRegressionExample("src/salary.csv"))
