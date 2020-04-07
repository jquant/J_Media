# Descrição de cada arquivo .py

## 1_Quantile_Regression.py
Este código roda um **Gradient Boosting Regressor(GBR)** com duas loss diferentes em dois conjuntos de dados diferentes, ou seja, temos 4 coisas dentro desse arquivo, são elas:
- 1) GBR com loss _ls_(least squares regression) e dados normais
- 2) GBR com loss _ls_(least squares regression) e dados com outlier sintético inserido
- 3) GBR com loss _lad_(leas absolute deviation) e dados normais
- 4) GBR com loss _lad_(leas absolute deviation) e dados com outlier sintético inserido
A ideia é comparar qual o efeito de cada loss em dados com e sem outliers. Após a comparação quantitativa, plotamos um gráfico para termos uma intuição do que está acontecendo no cenário que temos outliers.