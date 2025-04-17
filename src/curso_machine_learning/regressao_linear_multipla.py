import preprocesso as pre
import pandas as pd
import numpy as np

# temporizador
import time
from functools import wraps

def computeAutomaticBackwardElimination(XTrain, yTrain, XTest, sl):
    import statsmodels.api as sm
    XTrain = np.insert(XTrain, 0, 1, axis=1)
    XTest = np.insert(XTest, 0, 1, axis=1)

    numVars = len(XTrain[0])
    for i in range(0, numVars):
        regressor_ols = sm.OLS(yTrain, XTrain.astype(float)).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_ols.pvalues[j].astype(float) == maxVar):
                    #print("Deletar coluna", j)
                    XTrain = np.delete(XTrain, j, 1)
                    XTest = np.delete(XTest, j, 1)

    #regressor_ols.summary()
    return XTrain, XTest

def computeMultiLinearRegression(XTrain, yTrain, XTest, yTest):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Criar o modelo de regressão linear
    regressor = LinearRegression()

    # Ajustar o modelo aos dados de treino
    regressor.fit(XTrain, yTrain)

    # Prever os resultados para o conjunto de teste
    yPred = regressor.predict(XTest)

    # Calcular o erro médio quadrático (MSE)
    mse = mean_squared_error(yTest, yPred)

    # Calcular o coeficiente de determinação (R²)
    r2 = r2_score(yTest, yPred)

    print(f"Erro médio quadrático: {mse}")
    print(f"Coeficiente de determinação: {r2}")

    # gerar gráficos
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.scatter(yTest, yPred, alpha=0.7)
    # Linha identidade para referência
    lims = [min(yTest.min(), yPred.min()), max(yTest.max(), yPred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Previsto')
    plt.title('Real vs Previsto')
    plt.tight_layout()
    plt.show()
    
def runMultipleLinearRegression(filename):
    X, y = pre.loadDataset(filename)
    X = pre.fillMissingData(X, 0, 2)
    X = pre.computeCategorization(X, 3)
    XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.2)
    XTrain, XTest = computeAutomaticBackwardElimination(XTrain, yTrain, XTest, 0.05)
    computeMultiLinearRegression(XTrain, yTrain, XTest, yTest)

if __name__ == "__main__":
    runMultipleLinearRegression("src/insurance.csv")