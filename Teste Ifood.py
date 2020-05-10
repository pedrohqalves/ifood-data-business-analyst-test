#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importando as libs necessárias para a EDA


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 1- Análise exploratória de dados

# In[2]:


#importando a base de dados, como não é uma base muito grande, vamos usar diretamente do repositório do Github


dados = pd.read_csv("https://raw.githubusercontent.com/ifood/ifood-data-business-analyst-test/master/ml_project1_data.csv")


# In[3]:


dados.describe()


# In[4]:


#verificando a quantiadde de dados nulos na tabela

dados.isna().sum()


# In[5]:


# apenas a coluna income tem dados nulos vamos verificar estes casos

dados[dados['Income'].isna() == True]


# In[6]:


# como são apenas 24 casos, vamos remover essas linhas. 

dados2 = dados.dropna()


# In[7]:


#Olhando a distribuição dos clientes por Nível de Educação

round(dados.groupby('Education').ID.count()/dados.ID.count()*100,2)

#A maior parte dos clientes possui graduação completa.


# In[8]:


plt.figure(figsize = (10,5))
sns.barplot(x = dados.Education, y=dados.Income, hue=dados.Response).set(
    title ='Receita Média por nível de Educação e Resposta à pesquisa final')

# É possível perceber que geralmente quem compra os produtos possui renda média superior a quem não compra


# In[9]:


dados.columns


# In[92]:


#Verificando o ano de nascimento dos clientes é possível perceber que a grande maioria dos clientes nasceu entre 1960 e 1980 
dados.Year_Birth.hist()


# In[32]:


dados['month_year'] = pd.to_datetime(dados['Dt_Customer']).dt.to_period('M')


# In[93]:


#Verificando também a data que os clientes se envolveram com a empresa
plt.figure(figsize = (20,5))
sns.countplot(x='month_year', data = dados, order= sorted(dados.month_year.unique())).set(title = 'Contagem de Clientes por ano de início do relacionamento com a empresa')

#Os clientes se envolveram desde o 2º semestre de 2012 até o 2º semestre de 2014 de uma forma bem distribuída
# Isso pode demonstrar um crescimento próximo de linear na captação da empresa


# In[94]:


#Verificando a mesma premissa da linha anterior colocando também a variável de resposta (target)

plt.figure(figsize = (20,5))
sns.countplot(x='month_year', data = dados, hue = 'Response', order= sorted(dados.month_year.unique())).set(
title = 'Contagem de Clientes por Ano de Relacionamento com a Empresa separados por Resposta da Última Campanha')

# É possível perceber que esta última campanha (target) foi mais popular entre os membros mais antigos deste produto


# In[95]:


#Dado que a maioria dos clientes já possui mais de 30 anos, vamos verificar a quantidade de filhos de cada cliente
pivot = dados.pivot_table(values = 'ID',index= 'Kidhome',columns = 'Teenhome',aggfunc=lambda x: x.count()/dados.ID.count()*100)
sns.heatmap(pivot, annot=True, fmt = '.3g').set(title='Heatmap de Crianças por Adolescentes')

#Temos > 60% dos clientes com pelo menos 1 filho (seja criança ou adolescente)
#Clientes com 2 filhos são mais raros


# In[84]:


#Vamos entender dentro das famílias com filhos como é o comportamento, para isso vamos criar uma coluna a mais
#Essa coluna irá dizer a quantidade de filhos na família (independente de criança ou adolescente)
dados['children'] = dados.Kidhome+dados.Teenhome

pivot = dados.pivot_table(values = 'ID', index = 'children', columns = 'Marital_Status', 
                          aggfunc= lambda x: x.count()/dados.ID.count()*100)

sns.heatmap(pivot, annot=True, fmt = '.3g').set(title='Heatmap de Status Civil por número de Crianças')

# É possível perceber que temos a maior parte dos dados relevantes (~86%) dos clientes divididos da seguinte forma
# Solteiros, Casados e Casais que moram juntos, dentro desta faixa a maioria possui apenas 1 filho
# Divorciados representam quase 10% do restante.


# In[13]:


#Vamos entender como é o consumo dos clientes

dados.groupby('children')[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2']].sum()/dados.ID.count()*100

# Para quem não tem filhos, a campanha 2 foi mais atrativa
# Para quem tem 1 filho, a campanha 5 foi mais atrativa
# Para quem tem 2 filhos a campanha 4 foi mais atrativa
# Para quem tem 3 filhos a campanha 3 e 5 foram mais atrativas


# In[14]:


dados.columns


# In[97]:


# Vamos entender se existe relação entre as reclamações dos clientes e a recência
# o objetivo é entender se o problema que gerou a reclamação fez com que o usuário parasse de usar o produto

sns.barplot(data=dados, x='Recency',y='Complain', orient='h', hue='Response').set(title = 'Reclamações x Recência')

#Um fato interessante que é possível perceber é que quem aceitou essa última campanha geralmente tem recência menor
# Isso quer dizer que a campanha atinge na maior parte os clientes que usam mais o produto 
# É possível perceber também que quem reclama tem recência maior que quem não reclama (valiadndo a hipótese que os clientes que ficam incomodados compram menos vezes)


# In[16]:


#É interessante entender qual o perfil dos clientes que aceitaram a última campanha, e as anteriores também.
#Para isso vamos entender qual tipo de produto cada cliente compra mais e também quanto cada cliente gasta no total

dados['spend'] = dados['MntWines']+dados['MntFruits']+dados['MntMeatProducts']+dados['MntFishProducts']+dados['MntSweetProducts']
dados['choice'] = pd.Series()
dadosprod = dados[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']]
dados.choice = dadosprod.apply(lambda x: dadosprod.idxmax(axis = 1))
dados.choice = dados.choice.apply(lambda x: x[3:] )


# In[82]:


#Vamos entender agora qual a recência dos clientes que gastam mais, para confirmar se os que gastam mais é porque compram mais vezes

dados.pivot_table(values = 'spend', index = 'Recency',  aggfunc='sum').plot(figsize=(15,5), title = 'Gasto Total por Recência')

#Olhando os clientes é possível perceber que mesmo aqueles que compram com maior frequência possuem um gasto semelhante aos demais


# In[96]:


# Analisando os tipos de produtos que são comprados pelos clientes mais frequentes

dados.pivot_table(values='Recency', index = 'choice', aggfunc='mean').plot(figsize =(10,5), title = 'Recência Média por Produto')

# Geralmente os clientes com maior frequência (menor recência) gastam mais com produtos doces e gastam menos com frutas


# In[ ]:




