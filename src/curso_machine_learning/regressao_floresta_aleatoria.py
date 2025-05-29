import preprocesso as pre
import numpy as np
import pandas as pd

#temporizador
import time
from functools import wraps

def computeRandomForestRegressionModel(X, y, numberOfTrees):
    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=numberOfTrees)
    regressor.fit(X,y)

    return regressor

def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color = 'red') #plot real y points
    plt.plot(XLine, yLine, color = 'blue') #plot predicted points in line
    plt.title("Comparando pontos reais com a reta produzida pela regressão floresta aleatória")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()

def runRandomForestRegressionModelExample(filename):
    start_time = time.time()
    X, y = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.fillMissingData(X, 0, 1)
    elapsed_time = time.time() - start_time
    print("Fill Missing Data: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    rfModel = computeRandomForestRegressionModel(X, y, 50)
    elapsed_time = time.time() - start_time
    print("Compute Random Forest Regression Model: %.2f" % elapsed_time, "segundos.")
    
    showPlot(X, y, X, rfModel.predict(X))

    XGrid = np.arange(min(X), max(X), 0.01)
    XGrid = XGrid.reshape((len(XGrid), 1))

    showPlot(X, y, XGrid, rfModel.predict(XGrid))

    from sklearn.metrics import r2_score
    return r2_score(y, rfModel.predict(X))

if __name__ == "__main__":
    print(runRandomForestRegressionModelExample("src/salary.csv"))
