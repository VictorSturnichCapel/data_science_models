import regressao_arvore_decisao as dt
import regressao_floresta_aleatoria as rf
import regressao_linear as rl
import regressao_linear_multipla as rlm
import regressao_polinomial as rp
import regressao_vetor_suporte as rvs
import preprocesso as pre

import pandas as pd
import numpy as np
import time
from functools import wraps

def evaluateAllRegressionModels(filename_movie, filename_salary):
    scoreLR = rl.run_LinearRegression(filename_movie)
    scorePR2 = rp.runPolynomialLinearRegressionExample(filename_salary, 2)
    scorePR3 = rp.runPolynomialLinearRegressionExample(filename_salary, 3)
    scorePR4 = rp.runPolynomialLinearRegressionExample(filename_salary, 4)
    scoreDT = dt.runDecisionTreeRegressionModelExample(filename_salary)
    scoreRF = rf.runRandomForestRegressionModelExample(filename_salary)
    print(scorePR4, scoreDT, scoreRF)

evaluateAllRegressionModels('src/movie_data.csv','src/salary.csv')