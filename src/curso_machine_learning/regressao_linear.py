import pandas as pd
import numpy as np

def loadDataset(filename):
    # Carregar os dados
    df = pd.read_csv(filename, sep=';')

    # variável independente
    X = df.iloc[:, :-1].values

    # variável dependente
    y = df.iloc[:, -1].values

    return X, y


def fillMissingDataX(X):
    from sklearn.impute import SimpleImputer
    # Criar o imputer, que irá transformar os dados faltantes em mediana
    #imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    # Ajustar o imputer aos dados
    imputer = imputer.fit(X[:, 1:])

    # Transformar os dados
    X[:, 1:] = imputer.transform(X[:, 1:])

    return X

def fillMissingDatay(y):
    from sklearn.impute import SimpleImputer
    # Criar o imputer, que irá transformar os dados faltantes em média
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Ajustar o imputer aos dados
    y = imputer.fit_transform(y.reshape(-1, 1)).ravel()

    return y

def computeCategorization(X):
    from sklearn.preprocessing import LabelEncoder
    # Mudar palavra para numeros (rótulos)
    # Cria um problema, pois valores com diferenças grandes
    labelenconder_X = LabelEncoder()
    X[:, 0] = labelenconder_X.fit_transform(X[:, 0])

    # Transformar os dados categóricos em variáveis dummy # one hot encoding
    D = pd.get_dummies(X[:, 0])
    # vamos retirar ela então
    X = X[:, 1:]
    X = np.insert(X, 0, D.values, axis=1)

    return X

def splitTrainTestSets(X, y, testSize):

    # Conjunto de treino e teste
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=0)
    
    return XTrain, XTest, yTrain, yTest

def computeScaling(train, test):
    # Normalização
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)

    return train, test

def computeLinearRegressionModel(XTrain, yTrain, XTest, yTest):
    # Regressão Linear simples, prever uma variável dependente a partir de uma variável independente
    from sklearn.linear_model import LinearRegression
    # Criar o modelo de regressão linear
    regressor = LinearRegression()
    # Treinar o modelo
    regressor.fit(XTrain, yTrain)

    return regressor

    # Prever os resultados
    yPred = regressor.predict(XTest)

    ''' Avaliar o modelo
    O Mean Squared Error (MSE), ou Erro Quadrático Médio em português, é uma métrica muito utilizada para avaliar a qualidade de modelos de aprendizado de máquina e estatística, principalmente em problemas de regressão.

    Definição Matemática
    O MSE calcula a média dos quadrados das diferenças entre os valores previstos pelo modelo e os valores reais (ground truth). A fórmula é:
    Interpretação

    Quanto menor o MSE, melhor o modelo, pois indica que as previsões estão mais próximas dos valores reais.

    Como ele eleva os erros ao quadrado, penaliza mais os erros grandes do que os pequenos.

    É uma métrica sempre não negativa (pois os erros são elevados ao quadrado).

    Comparação com Outras Métricas
    MAE (Mean Absolute Error): Média do valor absoluto dos erros (menos sensível a outliers).

    RMSE (Root Mean Squared Error): Raiz quadrada do MSE (tem a mesma unidade da variável original). '''

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(yTest, yPred)
    print(f'yTest is: {yTest} and yPred is: {yPred}')
    print(f"Mean Squared Error: {mse}")

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

def run_LinearRegression(filename):
    X, y = loadDataset(filename)
    X = fillMissingDataX(X)
    y = fillMissingDatay(y)
    X = computeCategorization(X)
    XTrain, XTest, yTrain, yTest = splitTrainTestSets(X, y, 0.2)
    # Para regressão linear é bom manter as escalas do jeito que veio
    #XTrain, XTest = computeScaling(XTrain, XTest)
    regressor = computeLinearRegressionModel(XTrain, yTrain, XTest, yTest)

    from sklearn.metrics import r2_score
    return r2_score(yTest, regressor.predict(XTest))

if __name__ == "__main__":
    print(run_LinearRegression('src/movie_data.csv'))
