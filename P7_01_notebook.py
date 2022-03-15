# %%
import os
import pandas as pd

pd.options.plotting.backend = "plotly"
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
# %%
write_data = True

# True : création d'un dossier Figures et Tableau
# dans lesquels seront créés les éléments qui serviront à la présentation
# et écriture des figures et tableaux dans ces dossier
#
# False : pas de création de dossier ni de figures ni de tableaux

if write_data is True:
    try:
        os.mkdir("./Figures/")
    except OSError as error:
        print(error)
    try:
        os.mkdir("./Tableaux/")
    except OSError as error:
        print(error)
else:
    print("""Visualisation uniquement dans le notebook
    pas de création de figures ni de tableaux""")
# %% [markdown]
# inspiration du kernel kaggle :
# https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
#
# Visualisation des liens entre les fichiers :
# ![](home_credit.png)
# %%
App_train = pd.read_csv('./application_train.csv')
App_test = pd.read_csv('./application_test.csv')
# %%
App_train.info()
# %%
App_test.info()
# %%
print('Il y a {} colonnes ayant des valeurs maquantes'.format(
    len(App_train.isna().sum()[App_train.isna().sum() != 0].sort_values())))
# %%
# graphique du nombre de données
fig = px.bar(
    x=App_train.isna().sum()[App_train.isna().sum() != 0].sort_values().index,
    y=(App_train.shape[0] -
       App_train.isna().sum()[App_train.isna().sum() != 0].sort_values().values
       ) / App_train.shape[0] * 100,
    labels=dict(x='Indicateurs', y='Pourcentage de données'),
    title='Pourcentage de données par colonnes',
    height=550,
    width=1100)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/Data_trainNbFull.pdf')
# %%
# nombre de catégories par colonnes catégorielles
App_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
# %%
# encodage des colonnes catégorielles à 2 catégories
le = LabelEncoder()
le_count = 0
col_enc = []
for col in App_train.select_dtypes('object'):
    if len(App_train[col].unique()) <= 2:
        le.fit(App_train[col])
        App_train[col] = le.transform(App_train[col])
        App_test[col] = le.transform(App_test[col])
        col_enc.append(col)
        le_count += 1

print('{} colonnes ont été encodées'.format(le_count))
print(col_enc)
# %%
# créations de "dummy variables" pour les colonnes ayant plus de 2 catégories
App_train = pd.get_dummies(App_train, dummy_na=True)
App_test = pd.get_dummies(App_test, dummy_na=True)
# %%
print('Dimensions train : {}'.format(App_train.shape))
print('Dimensions test : {}'.format(App_test.shape))
# %%
App_train, App_test = App_train.align(App_test, axis=1, fill_value=0)
App_test.drop(columns='TARGET', inplace=True)
# %%
print('Dimensions train : {}'.format(App_train.shape))
print('Dimensions test : {}'.format(App_test.shape))
# %%
# description des ages des clients en années
(App_train.DAYS_BIRTH / -365).describe()
# %%
# description du temps ou le client a un emploi
(App_train.DAYS_EMPLOYED / -365).describe()
# %%
# la valeur -1000 semble aberrante
AnomEmploi = App_train[App_train.DAYS_EMPLOYED ==
                       App_train.DAYS_EMPLOYED.max()]
NAnomEmploi = App_train[
    App_train.DAYS_EMPLOYED != App_train.DAYS_EMPLOYED.max()]
print("Il y a {} clients ayant un temps d'emploi anormal".format(
    len(AnomEmploi)))
print(
    "Les clients ayant un temps d'emploi anormal ont {}% de défauts sur leurs emprunts"
    .format(round(100 * AnomEmploi.TARGET.mean(), 2)))
print(
    "Les clients ayant un temps d'emploi normal ont {}% de défauts sur leurs emprunts"
    .format(round(100 * NAnomEmploi.TARGET.mean(), 2)))
# %%
# création d'un flag anomalie
App_train[
    'DAYS_EMPLOYED_ANOM'] = App_train.DAYS_EMPLOYED == App_train.DAYS_EMPLOYED.max(
    )
App_train.DAYS_EMPLOYED_ANOM = App_train.DAYS_EMPLOYED_ANOM.astype(int)
App_train.DAYS_EMPLOYED.replace({App_train.DAYS_EMPLOYED.max(): np.nan},
                                inplace=True)
# %%
App_test.DAYS_EMPLOYED.describe()
# %%
# même problème pour les données test
# création d'un flag anomalie
App_test[
    'DAYS_EMPLOYED_ANOM'] = App_test.DAYS_EMPLOYED == App_test.DAYS_EMPLOYED.max(
    )
App_test.DAYS_EMPLOYED_ANOM = App_test.DAYS_EMPLOYED_ANOM.astype(int)
App_test.DAYS_EMPLOYED.replace({App_test.DAYS_EMPLOYED.max(): np.nan},
                               inplace=True)
# %%
px.histogram(
    App_train.DAYS_EMPLOYED / -365,
    labels={
        'value': "Années d'emploi"
    },
    title=
    "Histogramme du nombre de clients en fonction<br>de leur nombre d'années d'emploi"
).show(renderer='notebook')
# %%
TargetCorr = App_train.corr().TARGET.sort_values()
# %%
print('Meilleures correlations positives :\n', TargetCorr.tail(20))
# %%
print('\nMeilleures correlations négatives :\n', TargetCorr.head(15))
# %% [markdown]
# Meilleure corrélation positive pour DAYS_BIRTH
# %%
# Visualisation du nombre de client ayant ou non fait défaut en fonction de leur age
fig = px.histogram(
    App_train,
    App_train.DAYS_BIRTH / -365,
    color=App_train.TARGET.map({
        1: 'défaut de paiement',
        0: 'pas de défaut de paiement'
    }),
    marginal='box',
    labels={'x': 'Age'},
    title=
    'Histogramme du nombre de clients ayant ou non <br>fait défaut en fonction de leur age'
)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/HistPayement.pdf')
# %%
# dataframe age
age_data = App_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / -365

# Bin age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],
                                  bins=np.linspace(20, 70, num=11))
age_data.head(10)
# %%
# Groupe par bin et moyenn
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups
# %%
# figure des défauts de paiements par catégories d'age
fig = px.bar(
    age_groups,
    age_groups.index.astype(str),
    age_groups.TARGET * 100,
    labels={
        'x': "Groupes d'ages",
        'y': 'Défaut de paiement (%)'
    },
    title="Pourcentage de défauts de paiement en fonction des catégories d'ages"
)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/HistDefAge.pdf')
# %% [markdown]
# meilleures corrélation négatives EXT_SOURCE
# %%
ext_data = App_train[[
    'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
]]
ExtCorr = ext_data.corr().round(2)
# %%
ExtCorr = ExtCorr.where(np.tril(np.ones(ExtCorr.shape)).astype('bool'))
# heatmap à partir de cette matrice
fig = px.imshow(ExtCorr, text_auto=True, color_continuous_scale='balance')
fig.update_layout(plot_bgcolor='white')
fig.update_coloraxes(showscale=False)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/HeatmapExt.pdf')

# %%
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    fig = px.histogram(
        App_train,
        App_train[source],
        color=App_train.TARGET.map({
            0: 'pas de défaut de paiement',
            1: 'défaut de paiement'
        }),
        marginal='box',
        labels={'x': source},
        title=
        'Histogramme du nombre de clients ayant ou non <br>fait défaut en fonction de {}'
        .format(source))
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/HistSource{}.pdf'.format(i + 1))
        fig = px.histogram()
# %%
