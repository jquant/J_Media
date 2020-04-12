"""

"""
# importando bilbiotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


# constantes
PATH_FILE = '../data/boston_housing.csv'
TEST_SIZE = 0.2 
RANDOM_SEED = 42 # Answer to the Ultimate Question of Life, The Universe, and Everything

# leitura dos dados
df_housing = pd.read_csv(PATH_FILE)

# separando entre features e labels
X = df_housing.drop(columns='MEDV')
y = df_housing.loc[:, 'MEDV']

# separando entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# instanciando com cada alpha
gbr_lower = GradientBoostingRegressor(random_state=RANDOM_SEED, loss='quantile', alpha=0.05)
gbr_median = GradientBoostingRegressor(random_state=RANDOM_SEED, loss='quantile', alpha=0.5)
gbr_upper = GradientBoostingRegressor(random_state=RANDOM_SEED, loss='quantile', alpha=0.95)

# lista modelos e nomes
list_models = [('Lower_Bound', gbr_lower),
                ('Median', gbr_median),
                ('Upper_Bound', gbr_upper)]

# dataframe vazio para inserirmos as previsoes dos dados de teste
df_test = pd.DataFrame()
df_test.loc[:, 'y_test'] = y_test

# treinando e fazendo previsoes para cada alpha
for name, model in list_models:
    model.fit(X_train, y_train)
    df_test.loc[:, name] = model.predict(X_test)

# ordenando index para plotar
df_test.sort_index(inplace=True)

# visualizacao dos dados
plt.plot(df_test.loc[:, 'y_test'], 'b-', marker='o')
plt.plot(df_test.loc[:, 'Median'], 'r--', marker='o', label='Pred', alpha=0.5)
plt.fill_between(df_test.index, df_test.loc[:, 'Lower_Bound'], df_test.loc[:, 'Upper_Bound'], 
                label='I.C. 90%', alpha=0.4)
plt.legend()
plt.show()

# vendo quantos estao fora do intervalo
df_test.loc[:, 'Out_Bound'] = ~df_test.y_test.between(df_test.Lower_Bound, df_test.Upper_Bound)
perc_out_bound = df_test.loc[:, 'Out_Bound'].mean()*100
print(f'Perc valores fora do I.C: {perc_out_bound:.2f}%')