"""
Script que exporta os dados do Boston Housing em formato .csv

Escolhemos os dados direto do Scikit-Learn, concatenamos tudo dentro de um
unico DataFrame e exportamos em formato .csv para quem preferir usar assim.

Deixaremos tudo numa página única da internet, assim as pessoas podem passar apenas
a URL para a função pd.read_csv() e ler diretamente os dados da internet 

"""

# importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

# criando
data = load_boston()

# separando entre features e targets
X = data['data']
y = data['target'].reshape(-1, 1) # fazendo o reshape para podermos colocar tudo dentro de um unico np.ndarray

# dados par criacao do DataFrame
dataframe = np.hstack((X, y)) # np.ndarray com todos os dados das features e do target MEDV
cols_names = list(data.feature_names) + ['MEDV'] # lista com os nomes das colunas

# criando dataframe
df_boston = pd.DataFrame(data=dataframe, columns=cols_names)

# exportando dataframe em formato csv
df_boston.to_csv(path_or_buf='data/boston_housing.csv', index=False)