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
