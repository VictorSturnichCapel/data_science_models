import preprocesso as pre
import numpy as np
import pandas as pd

#temporizador
import time
from functools import wraps

def computeDecisionTreeRegressionModel(X, y):
    from sklearn.tree import DecisionTreeRegressor

    regressor = DecisionTreeRegressor()
    regressor.fit(X,y)

    return regressor

def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color = 'red') #plot real y points
    plt.plot(XLine, yLine, color = 'blue') #plot predicted points in line
    plt.title("Comparando pontos reais com a reta produzida pela regressão árvore de decisão")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()

def runDecisionTreeRegressionModelExample(filename):
    start_time = time.time()
    X, y = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.fillMissingData(X, 0, 1)
    elapsed_time = time.time() - start_time
    print("Fill Missing Data: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    dtModel = computeDecisionTreeRegressionModel(X, y)
    elapsed_time = time.time() - start_time
    print("Compute Decision Tree Regression Model: %.2f" % elapsed_time, "segundos.")
    
    showPlot(X, y, X, dtModel.predict(X))

    XGrid = np.arange(min(X), max(X), 0.01)
    XGrid = XGrid.reshape((len(XGrid), 1))

    showPlot(X, y, XGrid, dtModel.predict(XGrid))

    from sklearn.tree import export_graphviz
    export_graphviz(dtModel, out_file='tree.dot', feature_names=['Experiência'])
    # visualizar árvore em: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFBuQnNMekpBbnBhajJZWUZvQ0RITU9ocU03d3xBQ3Jtc0tscFBKNUEtWEprTnRCT0wwYTJ0eU1hN21KUTN4VGpLdS00TXl1c29xUjB6SWU3VEN0LUNfWU1KdGlUQ1k0MVNwbEdqdE56azYyN3YweHpBbEFsb0dITV94SU5WR1VZWldxR202VFlaR1FVcXhqMnFZTQ&q=https%3A%2F%2Fdreampuf.github.io%2FGraphvizOnline%2F&v=JwJcb-raZzo

if __name__ == "__main__":
    runDecisionTreeRegressionModelExample("src/salary.csv")
