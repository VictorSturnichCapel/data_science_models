import preprocesso as pre
import numpy as np
import pandas as pd

#temporizador
import time
from functools import wraps

def computePolynomialLinearRegressionModel(X, y, d):
    from sklearn.preprocessing import PolynomialFeatures
    polynomialFeatures = PolynomialFeatures(degree = d)
    XPolynomial = polynomialFeatures.fit_transform(X)
    
    from sklearn.linear_model import LinearRegression
    polyLinearRegression = LinearRegression()
    polyLinearRegression.fit(XPolynomial, y)

    return XPolynomial, polyLinearRegression

def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color = 'red') #plot real y points
    plt.plot(XLine, yLine, color = 'blue') #plot predicted points in line
    plt.title("Comparando pontos reais com a reta produzida pela regressão polinomial")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()

def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color = 'red') #plot real y points
    plt.plot(XLine, yLine, color = 'blue') #plot predicted points in line
    plt.title("Comparando pontos reais com a reta produzida pela regressão polinomial")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()


def runPolynomialLinearRegressionExample(filename, degree):
    start_time = time.time()
    X, y = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.fillMissingData(X, 0, 1)
    elapsed_time = time.time() - start_time
    print("Fill Missing Data: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    XPoly, polyLinearRegressor = computePolynomialLinearRegressionModel(X, y, degree)
    elapsed_time = time.time() - start_time
    print("Compute Polynomial Linear Regression: %.2f" % elapsed_time, "segundos.")
    
    showPlot(X, y, X, polyLinearRegressor.predict(XPoly))

    from sklearn.metrics import r2_score
    return r2_score(y, polyLinearRegressor.predict(XPoly))

if __name__ == "__main__":
    print(runPolynomialLinearRegressionExample("src/salary.csv"))
