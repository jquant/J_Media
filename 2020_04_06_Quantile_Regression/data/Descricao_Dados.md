# Descrição dos Dados

Os dados foram baixados direto do módulo [sklearn.datasets.load_boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn-datasets-load-boston). Exportamos os dados em .csv para quem prefira baixar e usar dessa maneira.<br>

<font color='red'>ATENÇÃO!:</font> a descrição abaixo não passa de uma tradução livre com alguns adicionais do próprio Scikit-Learn<br>

## Data Set Characteristics:

    :Número de samples(linhas): 506<br>

    :Número de features: 13 colunas numéricas/categóricas, que normalmente são a **Design Matrix**, ou seja, a tabela de Features
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
    
    :Target: MEDV, valor mediano da casa em questão
        - MEDV     Median value of owner-occupied homes in $1000's

    :Valores faltantes: Nenhum

    :Criador: Harrison, D. and Rubinfeld, D.L.<br>

Esta é uma cópia do dataset da UCI ML Housing.<br>
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/<br>

This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.<br>

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management,vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics...', Wiley, 1980.   N.B. Various transformations are used in the table on pages 244-261 of the latter.<br>

    : References

        - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
        - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.