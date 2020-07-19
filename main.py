#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[92]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from loguru import logger
from IPython import get_ipython
import imp
import sys


# In[73]:


def load_fifa(file):
    fifa = pd.read_csv(file)
    columns_to_drop = [
        "Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
        "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
        "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
        "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
        "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
        "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
        "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
        "CB", "RCB", "RB", "Release Clause"
    ]
    fifa.drop(columns_to_drop, axis=1, inplace=True)
    return fifa    


# In[74]:


fifa = load_fifa("fifa.csv")


# ## Inicia sua análise a partir daqui

# In[75]:


fifa.head()


# In[76]:


fifa.dtypes


# In[77]:


fifa.describe()


# In[78]:


missing = (fifa.isnull().mean() * 100).to_frame(name='missing(%)')
missing[missing['missing(%)'] > 0]


# In[79]:


fifa.shape


# In[80]:


# Apagar as linhas com dados faltantes: são apenas 0,26%
rows_before = fifa.shape[0]
fifa.dropna(axis=0, inplace=True)
f'Foram apagadas {rows_before - fifa.shape[0]} observações'


# In[81]:


# Há algum nulo ainda?
fifa.isnull().sum().any()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[82]:


def q1():
    pca_first_component = PCA(n_components=1).fit(fifa)    
    return float(np.round(pca_first_component.explained_variance_ratio_, 3))
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[83]:


def q2():
    return PCA(0.95).fit(fifa).explained_variance_ratio_.size
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[84]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[85]:


def q3():
    pca_coor = PCA(n_components = 2).fit(fifa)
    return tuple(np.round(pca_coor.components_.dot(x),3))
q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[86]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def q4():
    rfe = RFE(LinearRegression(), n_features_to_select = 5, step = 1)
    
    y_train = fifa['Overall'].copy()
    x_train = fifa.drop('Overall', axis=1)
    
    is_selected = rfe.fit(
        x_train, 
        y_train
    ).get_support()
    
    return list(x_train.columns[is_selected])
q4()

