"""
Script que roda um GradientBoostingRegressor com as duas loss
least squares e least absolute deviation, que é semelhante a usar a loss
quantile com alpha = 0.5

1.1) Rodaremos o estimador com as duas loss nos dados de treino normais e iremos
compará-los com scoring Root Mean Squared Error, depois iremos fazer previsão nos dados de teste e ver como os 
dois se saíram em dados que nunca viram

1.2) Rodaremos o estimador com as duas loss nos dados de treino, porém iremos inserir outliers de propósito
para ver qual o comportamento do estimador com cada loss, iremos compará-los através do scoring
Root Mean Squared Error, depois iremos fazer previsões nos dados de teste e ver como cada um se comporta nesse cenário
com outliers
"""
# importando bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# constantes
PATH_FILE = '../data/boston_housing.csv'
TEST_SIZE = 0.2 # porcentagem total dos dados que usaremos para teste
RANDOM_SEED = 42 # Answer to the Ultimate Question of Life, The Universe, and Everything
N_SPLITS = 10 # quantidade de folders do cross-validation
LIST_LOSS = [('Least_Squares', 'ls'), ('Least_Abs_Dev', 'lad'), ('Huber', 'huber')] # lista de tuplas com os nomes da loss e a loss
SCORING = 'neg_root_mean_squared_error' # metrica que usaremos dentro do cross-validation

# leitura dos dados
df_housing = pd.read_csv(PATH_FILE)

# separando entre features e labels
X = df_housing.drop(columns='MEDV')
y = df_housing.loc[:, 'MEDV']

# separando entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# criando um outlier nos dados de treino
y_train_outlier = y_train.copy()
y_train_outlier.iloc[0] = y_train_outlier.iloc[0] * 100

# lista com targets normais e alterados
list_targets = [('Target_Normal', y_train),
                ('Target_Alterado', y_train_outlier)]

# instanciando o kfold para o cross-validation
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# iterando na lista com targets normais e alterados
for target, y in list_targets:
    
    print('\n\n')
    print('----------------------------------------')
    print(f'RESULTADOS NO {target}')
    
    # iterando na lista com as duas loss que usaremos
    for name, loss in LIST_LOSS:
        # instanciando estimador e rodando o cross-validation
        model = GradientBoostingRegressor(loss=loss, random_state=RANDOM_SEED)
        scores = -cross_val_score(model, X_train, y, cv=kfold, scoring=SCORING)
        
        # treinando e depois prevendo nos dados de teste
        model.fit(X_train, y)
        y_pred = model.predict(X_test)
        
        # printando os resultado do cross-validation e depois nos dados de teste
        print(f'Resultado {name} no cross-validation com {N_SPLITS} splits')
        print(f'Média RMSE   : {scores.mean():.2f}')
        print(f'Desvio Padrão: {scores.std():.2f}')
        print(f'RMSE teste   : {mean_squared_error(y_test, y_pred):.2f}')
        print('----------')
        
#################################################################################
############ VISUALIZAÇÃO DO ESTIMADOR EM DADOS COM OUTLIER #####################
#################################################################################

