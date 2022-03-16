# %%
import os

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline
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
# %% [markdown]
# inspiration du kernel kaggle :
# https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
#
# Visualisation des liens entre les fichiers :
# ![](home_credit.png)
# %%
app_train = pd.read_csv('./application_train.csv')
app_test = pd.read_csv('./application_test.csv')
# %%
app_train.info()
# %%
app_test.info()
# %%
print('Il y a {} colonnes ayant des valeurs maquantes'.format(
    len(app_train.isna().sum()[app_train.isna().sum() != 0].sort_values())))
# %%
# graphique du nombre de données
fig = px.bar(
    x=app_train.isna().sum()[app_train.isna().sum() != 0].sort_values().index,
    y=(app_train.shape[0] -
       app_train.isna().sum()[app_train.isna().sum() != 0].sort_values().values
       ) / app_train.shape[0] * 100,
    labels=dict(x='Indicateurs', y='Pourcentage de données'),
    title='Pourcentage de données par colonnes',
    height=550,
    width=1100)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/Data_trainNbFull.pdf')
# %%
# nombre de catégories par colonnes catégorielles
app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
# %%
# encodage des colonnes catégorielles à 2 catégories
le = LabelEncoder()
le_count = 0
col_enc = []
for col in app_train.select_dtypes('object'):
    if len(app_train[col].unique()) <= 2:
        le.fit(app_train[col])
        app_train[col] = le.transform(app_train[col])
        app_test[col] = le.transform(app_test[col])
        col_enc.append(col)
        le_count += 1

print('{} colonnes ont été encodées'.format(le_count))
print(col_enc)
# %%
# créations de "dummy variables" pour les colonnes ayant plus de 2 catégories
app_train = pd.get_dummies(app_train, dummy_na=True)
app_test = pd.get_dummies(app_test, dummy_na=True)
# %%
print('Dimensions train : {}'.format(app_train.shape))
print('Dimensions test : {}'.format(app_test.shape))
# %%
app_train, app_test = app_train.align(app_test, axis=1, fill_value=0)
app_test.drop(columns='TARGET', inplace=True)
# %%
print('Dimensions train : {}'.format(app_train.shape))
print('Dimensions test : {}'.format(app_test.shape))
# %%
# description des ages des clients en années
(app_train.DAYS_BIRTH / -365).describe()
# %%
# description du temps ou le client a un emploi
(app_train.DAYS_EMPLOYED / -365).describe()
# %%
# la valeur -1000 semble aberrante
ano_emploi = app_train[app_train.DAYS_EMPLOYED ==
                       app_train.DAYS_EMPLOYED.max()]
Nano_emploi = app_train[
    app_train.DAYS_EMPLOYED != app_train.DAYS_EMPLOYED.max()]
print("Il y a {} clients ayant un temps d'emploi anormal".format(
    len(ano_emploi)))
print(
    "Les clients ayant un temps d'emploi anormal ont {}% de défauts sur leurs emprunts"
    .format(round(100 * ano_emploi.TARGET.mean(), 2)))
print(
    "Les clients ayant un temps d'emploi normal ont {}% de défauts sur leurs emprunts"
    .format(round(100 * Nano_emploi.TARGET.mean(), 2)))
# %%
# création d'un flag anomalie
app_train[
    'DAYS_EMPLOYED_ANOM'] = app_train.DAYS_EMPLOYED == app_train.DAYS_EMPLOYED.max(
    )
app_train.DAYS_EMPLOYED_ANOM = app_train.DAYS_EMPLOYED_ANOM.astype(int)
app_train.DAYS_EMPLOYED.replace({app_train.DAYS_EMPLOYED.max(): np.nan},
                                inplace=True)
# %%
app_test.DAYS_EMPLOYED.describe()
# %%
# même problème pour les données test
# création d'un flag anomalie
app_test[
    'DAYS_EMPLOYED_ANOM'] = app_test.DAYS_EMPLOYED == app_test.DAYS_EMPLOYED.max(
    )
app_test.DAYS_EMPLOYED_ANOM = app_test.DAYS_EMPLOYED_ANOM.astype(int)
app_test.DAYS_EMPLOYED.replace({app_test.DAYS_EMPLOYED.max(): np.nan},
                               inplace=True)
# %%
px.histogram(
    app_train.DAYS_EMPLOYED / -365,
    labels={
        'value': "Années d'emploi"
    },
    title=
    "Histogramme du nombre de clients en fonction<br>de leur nombre d'années d'emploi"
).show(renderer='notebook')
# %%
target_corr = app_train.corr().TARGET
# %%
print('Meilleures correlations positives :\n',
      target_corr.sort_values(ascending=False).head(10))
# %%
print('\nMeilleures correlations négatives :\n',
      target_corr.sort_values().head(10))
# %% [markdown]
# Meilleure corrélation positive pour DAYS_BIRTH
# %%
# Visualisation du nombre de client ayant ou non fait défaut en fonction de leur age
fig = px.histogram(
    app_train,
    app_train.DAYS_BIRTH / -365,
    color=app_train.TARGET.map({
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
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
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
# %%
ext_data = app_train[[
    'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
]]
ext_corr = ext_data.corr().round(2)
# %%
ext_corr = ext_corr.where(np.tril(np.ones(ext_corr.shape)).astype('bool'))
# visu matrice confusion
fig = px.imshow(ext_corr, text_auto=True, color_continuous_scale='balance')
fig.update_layout(plot_bgcolor='white')
fig.update_coloraxes(showscale=False)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CMExt.pdf')

# %%
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    fig = px.histogram(
        app_train,
        app_train[source],
        color=app_train.TARGET.map({
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
ext_data.isna().sum()
# %%
imputer = SimpleImputer(strategy='median')
trainfeatures = app_train[[
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
]]
testfeatures = app_test[[
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
]]

trainfeatures = imputer.fit_transform(trainfeatures)
testfeatures = imputer.fit_transform(testfeatures)
# %%
pol_trans = PolynomialFeatures(degree=3)
pol_trans.fit(trainfeatures)

trainfeat_trans = pol_trans.transform(trainfeatures)
testfeat_trans = pol_trans.transform(testfeatures)

# %%
trainfeat_transDF = pd.DataFrame(trainfeat_trans,
                                 columns=pol_trans.get_feature_names_out([
                                     'EXT_SOURCE_1', 'EXT_SOURCE_2',
                                     'EXT_SOURCE_3', 'DAYS_BIRTH'
                                 ]))
trainfeat_transDF = trainfeat_transDF.join(
    app_train.drop(
        columns={'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
                 }))
testfeat_transDF = pd.DataFrame(testfeat_trans,
                                columns=pol_trans.get_feature_names_out([
                                    'EXT_SOURCE_1', 'EXT_SOURCE_2',
                                    'EXT_SOURCE_3', 'DAYS_BIRTH'
                                ]))
testfeat_transDF = testfeat_transDF.join(
    app_test.drop(
        columns={'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
                 }))
# %%
trainfeat_corr = trainfeat_transDF.corr().TARGET
print('Meilleures correlations positives :\n',
      trainfeat_corr.sort_values(ascending=False).head(10))
# %%
print('\nMeilleures correlations négatives :\n',
      trainfeat_corr.sort_values().head(10))


# %%
def preprocessing(train, test):
    # scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_clean = train.drop(columns='TARGET')
    # impute
    imputer.fit(train_clean)
    train_imp = imputer.transform(train_clean)
    test_imp = imputer.transform(test)
    # scale
    scaler.fit(train_imp)
    train_scal = scaler.transform(train_imp)
    test_scal = scaler.transform(test_imp)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        train_scal, train["TARGET"])

    return (X_train, X_test, y_train, y_test)


# %%
X_train, X_test, y_train, y_test = preprocessing(app_train, app_test)
# %%
# baseline : logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=0)
log_reg.fit(X_train, y_train)
# %%
# predict_prob : 1ère colonne proba target = 0, seconde colonne proba target = 1
log_reg_pred = log_reg.predict(X_test)
log_reg_proba = log_reg.predict_proba(X_test)
print('Accuracy : {}'.format(round(log_reg.score(X_test, y_test), 2)))
# %%
fpr, tpr, thresholds = roc_curve(y_test, log_reg_proba[:, 1])
# %%
fig = px.area(x=fpr,
              y=tpr,
              title='ROC Curve (AUC={})'.format(round(auc(fpr, tpr), 4)),
              labels=dict(x='False Positive Rate', y='True Positive Rate'))
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
fig.show(renderer='notebook')
# %%
# matrice de confusion
CM = confusion_matrix(y_test, log_reg_pred)
CMfig = px.imshow(
    CM,
    x=['Pas de défaut<br>de paiement', 'Défaut de<br>paiement'],
    y=['Pas de défaut<br>de paiement', 'Défaut de<br>paiement'],
    text_auto=True,
    color_continuous_scale='balance',
    labels={
        'x': 'Catégorie prédite',
        'y': 'Catégorie réelle',
        'color': 'Nb clients'
    },
    title=
    'Matrice de confusion de la classification<br>par régression logistique')
CMfig.update_layout(plot_bgcolor='white')
CMfig.update_coloraxes(showscale=False)
CMfig.show(renderer='notebook')
if write_data is True:
    CMfig.write_image('./Figures/CMLogReg.pdf')

# %% [markdown]
# la majorité des clients sont classés comme ne faisant pas défaut
# %%
print(
    "Probabilité qu'un client ait fait défaut dans les données d'entrainement : {}"
    .format(
        round(app_train[app_train.TARGET == 1].shape[0] / app_train.shape[0],
              3)))

# %%
print(
    "Probabilité qu'un client ait fait défaut dans les données prédites : {}".
    format(
        round(log_reg_pred[log_reg_pred == 1].shape[0] / log_reg_pred.shape[0],
              3)))

# %% [markdown]
# il est difficile de bien prédire les clients faisant défaut du fait qu'ils sont
# peu nombreux par rapport à l'ensemble des clients
#
# Rééchantillonnage avec imblearn
# %%
# utilisation d'un échantillon de 30000 clients
X_train, X_test, y_train, y_test = preprocessing(
    app_train.sample(30000, random_state=0), app_test)
# %%
fig = go.Figure()
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

for samp in [
        RandomUnderSampler(random_state=0),
        TomekLinks(n_jobs=-1),
        RandomOverSampler(random_state=0),
        SMOTE(random_state=0, n_jobs=-1),
        SMOTEENN(random_state=0, n_jobs=-1),
        SMOTETomek(random_state=0, n_jobs=-1)
]:

    log_reg_samp = Pipeline([('sampler', samp),
                             ('logreg',
                              LogisticRegression(max_iter=1000,
                                                 random_state=0))])
    log_reg_samp.fit(X_train, y_train)
    log_reg_samp_pred = log_reg_samp.predict(X_test)
    log_reg_samp_proba = log_reg_samp.predict_proba(X_test)
    print('Accuracy : {}'.format(round(log_reg_samp.score(X_test, y_test), 2)))

    fpr, tpr, thresholds = roc_curve(y_test, log_reg_samp_proba[:, 1])
    name = '{} (AUC={})'.format(
        str(samp).split('(')[0], round(auc(fpr, tpr), 4))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    # matrice de confusion
    CM = confusion_matrix(y_test, log_reg_samp_pred)
    CMfig = px.imshow(
        CM,
        x=['Pas de défaut<br>de paiement', 'Défaut de<br>paiement'],
        y=['Pas de défaut<br>de paiement', 'Défaut de<br>paiement'],
        text_auto=True,
        color_continuous_scale='balance',
        labels={
            'x': 'Catégorie prédite',
            'y': 'Catégorie réelle',
            'color': 'Nb clients'
        },
        title=
        'Matrice de confusion de la classification<br>par régression logistique<br>et rééchantillonnage {}'
        .format(str(samp).split('(')[0]))

    CMfig.update_layout(plot_bgcolor='white')
    CMfig.update_coloraxes(showscale=False)
    CMfig.show(renderer='notebook')
    if write_data is True:
        CMfig.write_image('./Figures/CM{}.pdf'.format(
            #str(classifier).split('(')[0],
            str(samp).split('(')[0]))

fig.update_layout(xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CurvesROClogreg.pdf')
# %%
# utilisation d'un échantillon de 1000 clients
X_train, X_test, y_train, y_test = preprocessing(
    app_train.sample(1000, random_state=0), app_test)
# %%
sampler = [
    RandomUnderSampler(random_state=0),
    TomekLinks(n_jobs=-1),
    RandomOverSampler(random_state=0),
    SMOTE(random_state=0, n_jobs=-1),
    SMOTEENN(random_state=0, n_jobs=-1),
    SMOTETomek(random_state=0, n_jobs=-1)
]

param_grid = [{
    'sampling': sampler,
    'classifier': [LogisticRegression(max_iter=1000, random_state=0)]
}, {
    'sampling': sampler,
    'classifier': [RandomForestClassifier(n_jobs=-1, random_state=0)],
    'classifier__n_estimators': [100, 500, 1000]
}, {
    'sampling': sampler,
    'classifier': [GradientBoostingClassifier(random_state=0)],
    'classifier__loss': ['deviance', 'exponential'],
    'classifier__n_estimators': [100, 500, 1000],
    'classifier__max_depth': [2, 4, 6]
}, {
    'sampling':
    sampler,
    'classifier': [
        XGBClassifier(tree_method='gpu_hist',
                      gpu_id=0,
                      objective='binary:logistic',
                      eval_metric='auc',
                      use_label_encoder=False,
                      random_state=0)
    ]
}, {
    'sampling':
    sampler,
    'classifier': [LGBMClassifier(objective='binary', random_state=0)],
    'classifier__boosting_type': ['gbdt', 'dart', 'rf', 'goss'],
    'classifier__n_estimators': [100, 500, 1000]
}]

classifier = Pipeline([('sampling', 'passthrough'),
                       ('classifier', 'passthrough')])
grid = GridSearchCV(classifier,
                    param_grid,
                    scoring={
                        'AUC': 'roc_auc_ovo_weighted',
                        'Accuracy': 'balanced_accuracy',
                        'Precision': 'precision_weighted',
                        'Recall': 'recall_weighted',
                        'F1': 'f1_weighted'
                    },
                    refit='AUC',
                    n_jobs=-1)
grid.fit(X_train, y_train)
grid_pred = grid.predict(X_test)
grid_proba = grid.predict_proba(X_test)
print('Accuracy : {}'.format(
    grid.cv_results_['mean_test_Accuracy'][grid.best_index_]))
print('AUC : {}'.format(
    grid.cv_results_['mean_test_AUC'][grid.best_index_]))
print('Precision : {}'.format(
    grid.cv_results_['mean_test_Precision'][grid.best_index_]))
print('Recall : {}'.format(
    grid.cv_results_['mean_test_Recall'][grid.best_index_]))
print('F1 : {}'.format(
    grid.cv_results_['mean_test_F1'][grid.best_index_]))
# %%
fpr, tpr, thresholds = roc_curve(y_test, grid_proba[:, 1])
# %%
fig = px.area(x=fpr,
              y=tpr,
              title='{}/{}<br>ROC Curve (AUC={})'.format(
                  str(grid.best_params_['classifier']).split('(')[0],
                  str(grid.best_params_['sampling']).split('(')[0],
                  round(auc(fpr, tpr), 4)),
              labels=dict(x='False Positive Rate', y='True Positive Rate'))
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CurveROCGridBest.pdf')
# %%
# matrice de confusion
CM = confusion_matrix(y_test, grid_pred)
CMfig = px.imshow(
    CM,
    x=['Pas de défaut<br>de paiement', 'Défaut de<br>paiement'],
    y=['Pas de défaut<br>de paiement', 'Défaut de<br>paiement'],
    text_auto=True,
    color_continuous_scale='balance',
    labels={
        'x': 'Catégorie prédite',
        'y': 'Catégorie réelle',
        'color': 'Nb clients'
    },
    title=
    'Matrice de confusion de la classification<br>par {}<br>et échantillonnage par {}'
    .format(
        str(grid.best_params_['classifier']).split('(')[0],
        str(grid.best_params_['sampling']).split('(')[0]))

CMfig.update_layout(plot_bgcolor='white')
CMfig.update_coloraxes(showscale=False)
CMfig.show(renderer='notebook')
if write_data is True:
    CMfig.write_image('./Figures/CMGridBest.pdf')
# %%