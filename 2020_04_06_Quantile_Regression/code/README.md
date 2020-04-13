# Descrição de cada arquivo .py

## 1_LossOutlier.py
Este script roda um **Gradient Boosting Regressor(GBR)** com três loss diferentes em dois conjuntos de dados diferentes, ou seja, temos 6 coisas dentro desse arquivo, são elas:
- 1) GBR com loss _ls_(least squares regression) e dados normais
- 2) GBR com loss _ls_(least squares regression) e dados com outlier sintético inserido
- 3) GBR com loss _lad_(leas absolute deviation) e dados normais
- 4) GBR com loss _lad_(leas absolute deviation) e dados com outlier sintético inserido
- 5) GBR com loss _huber_ e dados normais
- 6) GBR com loss _huber_ e dados com outlier sintético inserido
A ideia é comparar qual o efeito de cada loss em dados com e sem outliers.

## 2_LossOutlierVis.py
Este script roda um **Gradient Boosting Regressor(GBR)** com três loss diferentes nos dados de treino. Inserimos outliers sintéticos nos dados de treino e criamos uma função para criar um scatter plot dos dados.

## 3_QuantileRegression.py
Este script roda um **Gradient Boosting Regressor(GBR)** com a loss quantile e dois alphas diferentes. Com isso, conseguimos montar previsões com um intervalo de 90% de confiança. Depois criamos um gráfico onde mostramos os dados
de teste e o Intervalo de Previsão, assim conseguimos analisar visualmente como o intervalo e as previsões se comportam
em torno da previsão com o alpha igual a 0.5(mediana). Por último, printamos na tela qual a porcentagem de dados que caem
fora do Intervalo de Previsão