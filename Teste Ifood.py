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

dados = dados.dropna()


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


# In[10]:


#Verificando o ano de nascimento dos clientes é possível perceber que a grande maioria dos clientes nasceu entre 1960 e 1980 
dados.Year_Birth.hist()


# In[11]:


dados['month_year'] = pd.to_datetime(dados['Dt_Customer']).dt.to_period('M')


# In[12]:


#Verificando também a data que os clientes se envolveram com a empresa
plt.figure(figsize = (20,5))
sns.countplot(x='month_year', data = dados, order= sorted(dados.month_year.unique())).set(title = 'Contagem de Clientes por ano de início do relacionamento com a empresa')

#Os clientes se envolveram desde o 2º semestre de 2012 até o 2º semestre de 2014 de uma forma bem distribuída
# Isso pode demonstrar um crescimento próximo de linear na captação da empresa


# In[13]:


#Verificando a mesma premissa da linha anterior colocando também a variável de resposta (target)

plt.figure(figsize = (20,5))
sns.countplot(x='month_year', data = dados, hue = 'Response', order= sorted(dados.month_year.unique())).set(
title = 'Contagem de Clientes por Ano de Relacionamento com a Empresa separados por Resposta da Última Campanha')

# É possível perceber que esta última campanha (target) foi mais popular entre os membros mais antigos deste produto


# In[14]:


#Dado que a maioria dos clientes já possui mais de 30 anos, vamos verificar a quantidade de filhos de cada cliente
pivot = dados.pivot_table(values = 'ID',index= 'Kidhome',columns = 'Teenhome',aggfunc=lambda x: x.count()/dados.ID.count()*100)
sns.heatmap(pivot, annot=True, fmt = '.3g').set(title='Heatmap de Crianças por Adolescentes')

#Temos > 60% dos clientes com pelo menos 1 filho (seja criança ou adolescente)
#Clientes com 2 filhos são mais raros


# In[15]:


#Vamos entender dentro das famílias com filhos como é o comportamento, para isso vamos criar uma coluna a mais
#Essa coluna irá dizer a quantidade de filhos na família (independente de criança ou adolescente)
dados['children'] = dados.Kidhome+dados.Teenhome

pivot = dados.pivot_table(values = 'ID', index = 'children', columns = 'Marital_Status', 
                          aggfunc= lambda x: x.count()/dados.ID.count()*100)

sns.heatmap(pivot, annot=True, fmt = '.3g').set(title='Heatmap de Status Civil por número de Crianças')

# É possível perceber que temos a maior parte dos dados relevantes (~86%) dos clientes divididos da seguinte forma
# Solteiros, Casados e Casais que moram juntos, dentro desta faixa a maioria possui apenas 1 filho
# Divorciados representam quase 10% do restante.


# In[16]:


#Vamos entender como é o consumo dos clientes

dados.groupby('children')[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2']].sum()/dados.ID.count()*100

# Para quem não tem filhos, a campanha 2 foi mais atrativa
# Para quem tem 1 filho, a campanha 5 foi mais atrativa
# Para quem tem 2 filhos a campanha 4 foi mais atrativa
# Para quem tem 3 filhos a campanha 3 e 5 foram mais atrativas


# In[17]:


dados.columns


# In[18]:


# Vamos entender se existe relação entre as reclamações dos clientes e a recência
# o objetivo é entender se o problema que gerou a reclamação fez com que o usuário parasse de usar o produto

sns.barplot(data=dados, x='Recency',y='Complain', orient='h', hue='Response').set(title = 'Reclamações x Recência')

#Um fato interessante que é possível perceber é que quem aceitou essa última campanha geralmente tem recência menor
# Isso quer dizer que a campanha atinge na maior parte os clientes que usam mais o produto 
# É possível perceber também que quem reclama tem recência maior que quem não reclama (valiadndo a hipótese que os clientes que ficam incomodados compram menos vezes)


# In[19]:


#É interessante entender qual o perfil dos clientes que aceitaram a última campanha, e as anteriores também.
#Para isso vamos entender qual tipo de produto cada cliente compra mais e também quanto cada cliente gasta no total

dados['spend'] = dados['MntWines']+dados['MntFruits']+dados['MntMeatProducts']+dados['MntFishProducts']+dados['MntSweetProducts']
dados['choice'] = pd.Series()
dadosprod = dados[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']]
dados.choice = dadosprod.apply(lambda x: dadosprod.idxmax(axis = 1))
dados.choice = dados.choice.apply(lambda x: x[3:] )


# In[20]:


#Vamos entender agora qual a recência dos clientes que gastam mais, para confirmar se os que gastam mais é porque compram mais vezes

dados.pivot_table(values = 'spend', index = 'Recency',  aggfunc='sum').plot(figsize=(15,5), title = 'Gasto Total por Recência')

#Olhando os clientes é possível perceber que mesmo aqueles que compram com maior frequência possuem um gasto semelhante aos demais


# In[21]:


# Analisando os tipos de produtos que são comprados pelos clientes mais frequentes

dados.pivot_table(values='Recency', index = 'choice', aggfunc='mean').plot(figsize =(10,5), title = 'Recência Média por Produto')

# Geralmente os clientes com maior frequência (menor recência) gastam mais com produtos doces e gastam menos com frutas


# In[22]:


#Verificando se temos outliers em alguma variável
dados.boxplot(figsize=(30,15))

#É possível perceber que na variável income temos outliers


# In[23]:


# Verificando a quantidade de outliers na variável income

(dados['Income']>150000).sum()

# Apenas 8 valores, vamos remover estes valores

dados = dados[dados['Income']<150000]


# # Vamos começar a segmentação dos clientes

# In[24]:


#Primeiro de tudo vamos transformar as variáveis categóricas para variáveis numéricas
(dados.dtypes == object).sum()

#Temos 4 variáveis categóricas


# In[25]:


dados.dtypes


# In[26]:


# Vamos seguir a seguinte estratégia
# Para a variáveis de data (Data de Cadastro do Cliente), vamos subtrair a data de hoje, transformando em um número para não perdermos a variável
# para a Educação, Status Civil, e Choice(produto preferido) vamos formar dummies
# para o mes de cadastro, vamos tirar a variável, ela foi criada para facilitar a visualização gráfica apenas

from datetime import datetime
dados['dtcadastro'] = dados.Dt_Customer.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
dados['dtcadastrado'] = datetime.today().date() - dados.dtcadastro
dados['dtcadastrado'] = dados.dtcadastrado.apply(lambda x: x.days)
dados2 = dados.drop(['month_year','Dt_Customer','dtcadastro'], axis=1)
dados2 = pd.get_dummies(dados2)


# In[27]:


# Indo para a padronização dos dados, ajustando a escala

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dados_scaled = scaler.fit_transform(dados2)


# In[28]:


# Primeiro vamos determinar a quantidade ideal de clusters para usarmos na segregação

from sklearn.cluster import KMeans
d_centroide = []
qtde = range(1,10)
for k in qtde:
    modelo = KMeans(n_clusters=k,random_state=42)
    modelo.fit(dados_scaled)
    d_centroide.append(modelo.inertia_)


# In[29]:


plt.figure(figsize=(10,5))

plt.plot(qtde, d_centroide, 'bx-')

plt.show()

# vamos usar 3 clusters para estabilizar a distância intra cluster (já decai 2/3 da distância aqui)


# In[77]:


# Aplicando o modelo
modelo = KMeans(n_clusters=3, random_state= 42)
modelo.fit(dados_scaled)


# In[78]:


# verificando os resultados

dados2['cluster'] = modelo.predict(dados_scaled)


# In[115]:


# Separando agora e analisando os clusters

dados2.groupby('cluster').Income.mean().plot(kind='barh')

# Em termos de Receita temos bastante diferença entre os clusters
# O Cluster 0 possui maior renda, seguido do cluster 2 e depois do cluster 1


# In[101]:


columns = dados2.columns


# In[121]:


fig, axes = plt.subplots(6,8,sharex=True, sharey=False, figsize = (30,30))
for ax, feature, name in zip(axes.flatten(), columns, columns):
    sns.scatterplot(data = dados2, x='ID',y=feature, ax=ax, hue = 'cluster')
    ax.set(title = name)
    
# Com os gráficos abaixo é possível entender melhor o comportamento dos clusters em cada variável.


# In[106]:


fig, axes = plt.subplots(5,5, sharex= True, sharey= False, figsize=(30,20))
dados2.boxplot(column=['Year_Birth','Income', 'Recency',
       'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
       'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact',
       'Z_Revenue', 'Response', 'children', 'dtcadastrado'], ax = axes, by='cluster')


# In[118]:


dados_without_income = dados2.drop('Income', axis = 1)


# In[119]:


plt.figure(figsize = (20,20))
sns.heatmap(dados_without_income.groupby('cluster').mean().transpose(),annot=True, fmt='.4g')

#Com o heatmap é possível ver de forma mais clara as diferenças entre os clusters de clientes.


# # Passando para a modelagem preditiva, para maximizar a margem de lucro da campanha

# In[122]:


dados2.columns


# In[132]:


# Assumiremos aqui o custo e retorno uniforme por cliente conforme as colunas Z_CostCustomer e Z_Revenue

#Começaremos pela redução das variáveis, de forma a começar o modelo com uma base mais enxuta

#Separando a base em features (x) e target(y)
# O target aqui será a resposta na campamnha (tentaremos atingir quem tem mais probabilidade de aceitar a campanha)

x = dados2.drop('Response', axis =1)
y = dados2.Response



# In[144]:


# Começando a feature selection, tentaremos com as 20 melhores variáveis para facilitar a interpretação do modelo.
print(x.shape)

from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=20)
x_new = selector.fit_transform(x,y)
cols = selector.get_support(indices=True)
x_best = x.iloc[:,cols]


print(x_new.shape)


# In[145]:


# Vamos entender quais foram as features mais interessantes para essa escolha

x_best.columns

# É possível perceber aqui que colunas agregadoras como spend e children foram escolhidas junto com as suas dependentes
# Pode ser que tenhamos redundância aqui, vamos fazer o teste depois sem estas variáveis para entender se temos melhores resultados.


# In[146]:


# Começando a modelagem

#Separando treino e teste

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x_best, y, test_size = 0.3, random_state = 42)


# In[147]:


# Começando com um modelo de Random Forest que seria um bom benchmark para comparação

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(x_treino,y_treino)

#base de treino
y_pred_treino_rf = rf.predict(x_treino)

#base de teste
y_pred_teste_rf = rf.predict(x_teste)


#Verificando o score do modelo
from sklearn.metrics import accuracy_score

score_treino_rf = round(accuracy_score(y_pred_treino_rf,y_treino) *100 , 2)
score_teste_rf = round(accuracy_score(y_pred_teste_rf, y_teste) *100 , 2)

print(" O Score do treino foi de "+ str(score_treino_rf) + '%')
print(' O Score do teste foi de '+str(score_teste_rf) + '%')

# Começamos com um bom score, de 85% para o teste. Vamos aplicar outras técnicas para garantir que teremos o melhor resultado


# In[148]:


#Regressão Logistica
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(x_treino,y_treino)

#base de treino
y_pred_treino_lr = lr.predict(x_treino)

#base de teste
y_pred_teste_lr = lr.predict(x_teste)


#Verificando o score do modelo
from sklearn.metrics import accuracy_score

score_treino_lr = round(accuracy_score(y_pred_treino_lr,y_treino) *100 , 2)
score_teste_lr = round(accuracy_score(y_pred_teste_lr, y_teste) *100 , 2)

print(" O Score do treino foi de "+ str(score_treino_lr) + '%')
print(' O Score do teste foi de '+str(score_teste_lr) + '%')

# A Regressão logística obteve resultados um pouco mais baixos aqui


# In[166]:


# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
gb.fit(x_treino,y_treino)

#base de treino
y_pred_treino_gb = gb.predict(x_treino)

#base de teste
y_pred_teste_gb = gb.predict(x_teste)


#Verificando o score do modelo
from sklearn.metrics import accuracy_score

score_treino_gb = round(accuracy_score(y_pred_treino_gb,y_treino) *100 , 2)
score_teste_gb = round(accuracy_score(y_pred_teste_gb, y_teste) *100 , 2)

print(" O Score do treino foi de "+ str(score_treino_gb) + '%')
print(' O Score do teste foi de '+str(score_teste_gb) + '%')

# O gradient boost se provou menos enviesado (um pouco menos preciso no treino mas mais preciso no teste)


# In[171]:


# Usando o Gradient Boosting, vamos melhorar os parâmetros do modelo usando Grid Search
from sklearn.model_selection import GridSearchCV
# dicionario do GB
parametros_gb = {
    
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"]
    
    }

modelo_gb = GridSearchCV(GradientBoostingClassifier(random_state = 42), parametros_gb, cv=5, n_jobs=-1,verbose=1)
modelo_gb.fit(x_treino, y_treino)


# In[172]:


modelo_gb.best_params_


# In[173]:


# Vamos conferir o score do modelo agora com os ajustes

#base de treino
y_pred_treino_gb_final = modelo_gb.predict(x_treino)

#base de teste
y_pred_teste_gb_final = modelo_gb.predict(x_teste)


#Verificando o score do modelo
from sklearn.metrics import accuracy_score

score_treino_gb_final = round(accuracy_score(y_pred_treino_gb_final,y_treino) *100 , 2)
score_teste_gb_final = round(accuracy_score(y_pred_teste_gb_final, y_teste) *100 , 2)

print(" O Score do treino foi de "+ str(score_treino_gb_final) + '%')
print(' O Score do teste foi de '+str(score_teste_gb_final) + '%')


# In[192]:


# Vamos entender qual a probabilidade de cada cliente aceitar a campanha

dados2['proba'] = modelo_gb.predict_proba(x_best)[:, 1]
dados2['predicaomodelo'] = modelo_gb.predict(x_best)
dados2['decisao'] = pd.Series()
dados2['aux'] = pd.Series()


# In[226]:


# Aqui vamos entender o ponto ótimo da campanha, menor erro e maior lucro, 
#a nossa decisão vai se basear na probabilidade do cliente ser convertido na campanha ou nao
# vamos tentar com probabilidades de 10 a 90%
custo_list=[]
venda_list = []
probabilidade_list = []
score_list = []
clientes_impactados_list = []
for i in range(1,10,1):
    probabilidade = i/10
    dados2['decisao'] = dados2.proba.apply(lambda x : 1 if x > probabilidade else 0)
    for j in dados2.index:
        dados2['aux'][j] = abs(dados2.Response[j] - dados2.decisao[j])
    score = (1 - (dados2.aux.sum()/dados2.ID.count())) *100
    clientes_impactados = dados2.decisao.sum() 
    custototal = clientes_impactados*dados2.Z_CostContact.mean()
    venda = score*clientes_impactados*dados2.Z_Revenue.mean()
    custo_list.append(custototal)
    venda_list.append(venda)
    probabilidade_list.append(probabilidade)
    score_list.append(score)
    clientes_impactados_list.append(clientes_impactados)
    print('loop' + str(i) + 'concluído')


# In[216]:


campanha = pd.DataFrame(columns={'probabilidade': probabilidade_list, 'venda': venda_list, 'custo': custo_list})


# In[217]:


campanha.probabilidade = probabilidade_list
campanha.venda = venda_list
campanha.custo = custo_list
campanha['lucro'] = campanha.venda-campanha.custo


# In[231]:


#Conseguimos ver que quanto maior a probabilidade que queremos usar para dizer que um cliente vai aceitar, menos clientes serão impactados
# Consequentemente 

plt.plot(clientes_impactados_list)
print(clientes_impactados_list)


# In[ ]:


# O Score do modelo é otimizado por volta de 40 a 50% de probabilidade, mas isso não quer dizer necessariamente que quanto maior o score, mais lucro
# Para isso devemos comparar quantos clientes estaremos atingindo e também qual o custo e lucro que teremos


# In[223]:


campanha.groupby('probabilidade').lucro.mean().plot(kind='barh')
# É possível ver que temos o maior lucro com 10% de probabilidade, impactando mais clientes.


# In[232]:


# Apesar de 10% ser uma probabilidade relativamente baixa, conseguimos explicar a maximização do lucro olhando o histograma

dados2.proba.hist()

# A maior parte dos clientes possuem probabilidades ainda menores que 10% de aceitar a campanha, e por isso não seriam impactados.


# In[244]:


print(('Com {} % de probabilidade teríamos {} de clientes impactados gerando um lucro de {} ').format(probabilidade_list[0]*100,clientes_impactados_list[0],lucro_list[0]))


# In[ ]:




