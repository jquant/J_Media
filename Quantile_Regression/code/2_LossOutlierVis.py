"""
Treinamos o Gradient Boosting Regressor com as 
loos ls, lad e huber nos dados com outliers sinteticos.

Colocamos uma funcao para criar um scatter plot, assim
podemos ver como cada loss se comporta nos dados de treino e teste.
Aqui plotamos somente a loss ls, sinta-se livre para plotar o que quiser
com a funcao plot_scatter().
"""
# importando bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


# funcoes auxiliares
def plot_scatter(X: pd.Series, y_1: pd.Series, y_2: pd.Series, 
                label_1: str, label_2: str, title: str, yscale: str):
    """
    Cria um scatter plot que compara uma pd.Series X com duas
    pd.Series y diferentes
    
    Arguments:
        X {pd.Series} -- valores de x para plotar
        y_1 {pd.Series} -- primeiro valor de y para plotar
        y_2 {pd.Series} -- segundo valor de y para plotar
        label_1 {str} -- label de y_1
        label_2 {str} -- label de y_2
        title {str} -- titulo do grafico
        yscale {str} -- escala de y={'linear', 'log', 'symlog', 'logit'}
    """
    plt.scatter(X, y_1, label=label_1)
    plt.scatter(X,y_2, label=label_2, alpha=0.6)
    plt.title(title)
    plt.yscale(value=yscale)
    plt.legend()
    plt.show()


# constantes
PATH_FILE = '../data/boston_housing.csv'
TEST_SIZE = 0.2 
RANDOM_SEED = 42
N_SPLITS = 10 
LIST_LOSS = [('Least_Squares', 'ls'), 
            ('Least_Abs_Dev', 'lad'), 
            ('Huber', 'huber')]
SCORING = 'neg_root_mean_squared_error'

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

# instanciando os modelos
gbr_ls = GradientBoostingRegressor(random_state=RANDOM_SEED, loss='ls')
gbr_lad = GradientBoostingRegressor(random_state=RANDOM_SEED, loss='lad')
gbr_hub = GradientBoostingRegressor(random_state=RANDOM_SEED, loss='huber')

# criando dataframe com os dados de treino e teste
df_train = pd.DataFrame()
df_train.loc[:, 'y_train'] = y_train_outlier

df_test = pd.DataFrame()
df_test.loc[:, 'y_test'] = y_test

# treinando os modelos e inserindo previsoes dos dados 
# de treino e de teste nos respectivos dataframes
for i, model in enumerate([gbr_ls, gbr_lad, gbr_hub]):
    model.fit(X_train, y_train_outlier)
    df_train.loc[:, LIST_LOSS[i][0]] = model.predict(X_train)
    df_test.loc[:, LIST_LOSS[i][0]] = model.predict(X_test)

# visualizacao dos dados de treino com outlier com a loss ls
plot_scatter(X=X_train.iloc[:, 0], y_1=y_train_outlier, y_2=df_train.loc[:, 'Least_Squares'], 
            label_1='Train_Real', label_2='Train_Pred', title='Dados de Treino', yscale='log')

# visualizacao dos dados de teste com a loss ls
plot_scatter(X=X_test.iloc[:, 0], y_1=y_test, y_2=df_test.loc[:, 'Least_Squares'], 
            label_1='Test_Real', label_2='Test_Pred', title='Reais vs Previsão', yscale='log')
