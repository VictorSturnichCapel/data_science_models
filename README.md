# Projeto de Regressão

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Este repositório contém três implementações de algoritmos de regressão utilizando Python e bibliotecas populares como `scikit-learn` e `matplotlib`.

---

## Arquivos

- **`regressao_linear.py`**  
  Regressão linear simples para modelar a relação entre uma variável independente e uma variável dependente!

- **`regressao_linear_multipla.py`**  
  Regressão linear múltipla usando várias variáveis independentes para prever um único valor!

- **`regressao_polinomial.py`**  
  Regressão polinomial para modelar relações não lineares nos dados.

---

## Pré-requisitos

Certifique-se de ter instalado:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Como Executar

Para executar qualquer script, use o terminal:

```bash
python regressao_linear.py
python regressao_linear_multipla.py
python regressao_polinomial.py
```

---

## Exemplos de Saída

### 📈 `regressao_linear.py`

Treina um modelo simples e plota o gráfico:

- **Saída no terminal:**
  ```
  Coeficiente angular (m): 2.35
  Intercepto (b): 4.76
  Score R²: 0.92
  ```

- **Gráfico gerado:**
  - Pontos de dados (scatter plot)
  - Linha de regressão ajustada sobre os dados

---

### 📊 `regressao_linear_multipla.py`

Treina um modelo com múltiplas variáveis:

- **Saída no terminal:**
  ```
  Coeficientes: [1.47, -2.12, 3.58]
  Intercepto: 5.31
  Score R²: 0.88
  ```

---

### 📈 `regressao_polinomial.py`

Aplica transformação polinomial e ajusta o modelo:

- **Saída no terminal:**
  ```
  Coeficientes do polinômio: [0.5, -1.3, 2.8, 0.1]
  Score R² (ajustado): 0.96
  ```

- **Gráfico gerado:**
  - Curva polinomial ajustada sobre os dados

---

## Estrutura dos Scripts

Cada arquivo segue o seguinte fluxo:

1. **Importação** das bibliotecas necessárias

2. **Criação** ou **carregamento** de dados de entrada
3. **Treinamento** do modelo
4. **Avaliação** do modelo
5. **Visualização** dos resultados (quando aplicável)

---
## Licença

Distribuído sob a licença MIT.  
Sinta-se livre para usar, modificar e compartilhar!



