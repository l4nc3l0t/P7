# %%
import os
import gc

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, \
    make_scorer, fbeta_score, precision_recall_curve, precision_score, recall_score, \
        f1_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline

import shap
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
# %% [markdown]
#### Fichier application_train/test.csv
# %%
app_train = pd.read_csv('./application_train.csv')
app_test = pd.read_csv('./application_test.csv')
# %%
app_train.info()
# %%
app_test.info()
# %%
print('Il y a {} colonnes ayant des valeurs maquantes'.format(
    len(app_train.isna().sum()[app_train.isna().sum() != 0])))
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
    fig.write_image('./Figures/app_trainNbDataMiss.pdf')
# %%
# suppression des lignes ayant des nans dans les colonnes ayant moins de 1 % de nan
app_train_clean = app_train[
    app_train.isna().sum()[(app_train.isna().sum() <= .01 * app_train.shape[0])
                           & (app_train.isna().sum() != 0)].index].dropna()
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
ext_data = app_train[target_corr.abs().sort_values(
    ascending=False).head(4).index.to_list()]
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
# %% [markdown]
# calcul des variables polynomiales à partir des variables les plus corrélées
# avec la cible
# %%
target_corr_extCol = target_corr.drop('TARGET').abs().sort_values(
    ascending=False).head(3).index.to_list()
trainfeatures = app_train[target_corr_extCol].set_index(app_train.index)

testfeatures = app_test[target_corr_extCol].set_index(app_test.index)
del target_corr
gc.collect()


# %%
def create_PolFeat(trainfeatures, testfeatures):
    pol_trans = PolynomialFeatures(degree=3)
    trainfeat_transDF = pd.DataFrame(index=trainfeatures.index)
    testfeat_transDF = pd.DataFrame(index=testfeatures.index)
    for o in range(len(trainfeatures.columns) - 1):
        for p in range(o + 1, len(trainfeatures.columns)):
            for c in [[o, p]]:

                def polfeat_trans(trainfeat_transDF, testfeat_transDF, c):
                    pol_trans.fit(trainfeatures.iloc[:, c].dropna())
                    trainfeat_trans = pol_trans.transform(
                        trainfeatures.iloc[:, c].dropna())
                    testfeat_trans = pol_trans.transform(
                        testfeatures.iloc[:, c].dropna())
                    trainfeat_transDF = trainfeat_transDF.join(
                        pd.DataFrame(
                            trainfeat_trans,
                            columns=pol_trans.get_feature_names_out(
                                trainfeatures.iloc[:, c].columns.to_list()),
                            index=trainfeatures.iloc[:,
                                                     c].dropna().index.to_list(
                                                     )),
                        rsuffix='_todrop')
                    testfeat_transDF = testfeat_transDF.join(pd.DataFrame(
                        testfeat_trans,
                        columns=pol_trans.get_feature_names_out(
                            testfeatures.iloc[:, c].columns.to_list()),
                        index=testfeatures.iloc[:,
                                                c].dropna().index.to_list()),
                                                             rsuffix='_todrop')
                    return (trainfeat_transDF, testfeat_transDF)

                trainfeat_transDF, testfeat_transDF = polfeat_trans(
                    trainfeat_transDF, testfeat_transDF, c)
            for q in range(p + 1, len(trainfeatures.columns)):
                for c in [[o, p, q]]:
                    trainfeat_transDF, testfeat_transDF = polfeat_trans(
                        trainfeat_transDF, testfeat_transDF, c)
    trainfeat_transDF = trainfeat_transDF.iloc[:, ~trainfeat_transDF.columns.
                                               str.endswith('_todrop')]
    testfeat_transDF = testfeat_transDF.iloc[:, ~testfeat_transDF.columns.str.
                                             endswith('_todrop')]
    return (trainfeat_transDF, testfeat_transDF)


# %%
trainfeat_transDF, testfeat_transDF = create_PolFeat(trainfeatures,
                                                     testfeatures)

# %%
trainfeat_transDF = trainfeat_transDF.join(
    app_train.drop(columns=target_corr_extCol).set_index(app_train.index))

testfeat_transDF = testfeat_transDF.join(
    app_test.drop(columns=target_corr_extCol).set_index(app_test.index))

# %%
trainfeat_corr = trainfeat_transDF.corr().TARGET
print('Meilleures correlations positives :\n',
      trainfeat_corr.sort_values(ascending=False).head(10))
# %%
print('\nMeilleures correlations négatives :\n',
      trainfeat_corr.sort_values().head(10))


# %% [markdown]
# Nous avons une légère amélioration de la corrélations de la cible
# avec certaines nouvelles variables
# %%
def preprocessing(train, test, imputerize=False):
    train_clean = train.drop(columns='TARGET')
    # impute
    if imputerize is True:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(train_clean)
        train_imp = imputer.transform(train_clean)
        test_imp = imputer.transform(test)
    else:
        train_imp = train_clean
        test_imp = test
    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_imp)
    train_scal = scaler.transform(train_imp)
    test_scal = scaler.transform(test_imp)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        train_scal, train["TARGET"])

    return (X_train, X_test, y_train, y_test)


# %%
X_train, X_test, y_train, y_test = preprocessing(app_train, app_test, True)
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
# utilisation d'un échantillon de 40000 clients
X_train, X_test, y_train, y_test = preprocessing(app_train.sample(
    40000, random_state=0),
                                                 app_test,
                                                 imputerize=True)
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


# %% [markdown]
# Avec TomekLink, la régression logistique classe tout les clients
# comme ne faisant pas défaut ce que l'on ne veut pas et SMOTENN,
# permet un classement naïf 50/50 qui ne reflète pas non plus la
# réalité
# %%
def GridPlot(classifier,
             param_grid,
             train,
             test,
             data_type=str,
             imputerize=True,
             refit='AUC',
             sample=None):

    if sample == None:
        X_train, X_test, y_train, y_test = preprocessing(train,
                                                         test,
                                                         imputerize=imputerize)
    else:
        X_train, X_test, y_train, y_test = preprocessing(train.sample(
            sample, random_state=0),
                                                         test,
                                                         imputerize=imputerize)

    Scores = pd.DataFrame()
    RocCurve = pd.DataFrame()
    feature_importance = pd.DataFrame()
    idx = 0
    IDX = 0

    Rocfig = go.Figure()
    Rocfig.add_shape(type='line',
                     line=dict(dash='dash'),
                     x0=0,
                     x1=1,
                     y0=0,
                     y1=1)

    for param_grid in param_grid:
        grid = GridSearchCV(classifier,
                            param_grid,
                            scoring={
                                'AUC': 'roc_auc_ovo_weighted',
                                'Accuracy': 'balanced_accuracy',
                                'Precision': 'precision_weighted',
                                'Recall': 'recall_weighted',
                                'F1': 'f1_weighted',
                                'F2': make_scorer(fbeta_score, beta=2)
                            },
                            refit=refit,
                            n_jobs=-1)
        grid.fit(X_train, y_train)
        grid_pred = grid.predict(X_test)
        grid_proba = grid.predict_proba(X_test)
        print(grid.best_estimator_)
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
        print('F2 : {}'.format(
            grid.cv_results_['mean_test_F2'][grid.best_index_]))

        fpr, tpr, thresholds = roc_curve(y_test, grid_proba[:, 1])
        name = '{} - {}<br>(AUC={})'.format(
            ''.join([
                c for c in str(grid.best_params_['classifier']).split('(')[0]
                if c.isupper()
            ]), ''.join([
                c for c in str(grid.best_params_['sampling']).split('(')[0]
                if c.isupper()
            ]), round(auc(fpr, tpr), 4))
        Rocfig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        RocCurve = pd.concat([
            RocCurve,
            pd.DataFrame({
                'Modèle':
                ''.join([
                    c
                    for c in str(grid.best_params_['classifier']).split('(')[0]
                    if c.isupper()
                ]) + '/' + ''.join([
                    c for c in str(grid.best_params_['sampling']).split('(')[0]
                    if c.isupper()
                ]) + '_' + data_type,
                'FPR':
                fpr,
                'TPR':
                tpr
            })
        ])
        RocCurve = RocCurve.reset_index().drop(columns={'index'})

        Scores = pd.concat([
            Scores,
            pd.DataFrame(
                {
                    'Modèle':
                    ''.join([
                        c for c in str(grid.best_params_['classifier']).split(
                            '(')[0] if c.isupper()
                    ]) + '/' + ''.join([
                        c for c in str(grid.best_params_['sampling']).split(
                            '(')[0] if c.isupper()
                    ]) + '_' + data_type,
                    'Accuracy':
                    grid.cv_results_['mean_test_Accuracy'][grid.best_index_],
                    'AUC':
                    grid.cv_results_['mean_test_AUC'][grid.best_index_],
                    'Precision':
                    grid.cv_results_['mean_test_Precision'][grid.best_index_],
                    'Recall':
                    grid.cv_results_['mean_test_Recall'][grid.best_index_],
                    'F1':
                    grid.cv_results_['mean_test_F1'][grid.best_index_],
                    'F2':
                    grid.cv_results_['mean_test_F2'][grid.best_index_]
                },
                index=[idx])
        ])
        idx += 1

        if str(grid.best_params_['classifier']).split(
                '(')[0] == 'LogisticRegression':
            feature_importance = pd.concat([
                feature_importance,
                pd.DataFrame(
                    {
                        'Modèle':
                        ''.join([
                            c for c in str(grid.best_params_['classifier']).
                            split('(')[0] if c.isupper()
                        ]) + '/' + ''.join([
                            c
                            for c in str(grid.best_params_['sampling']).split(
                                '(')[0] if c.isupper()
                        ]) + '_' + data_type,
                        'Feature':
                        test.columns,
                        'value':
                        grid.best_estimator_.named_steps['classifier'].coef_[0]
                    },
                    index=pd.RangeIndex(IDX, IDX + len(test.columns)))
            ])
            IDX += len(test.columns)
        else:
            feature_importance = pd.concat([
                feature_importance,
                pd.DataFrame(
                    {
                        'Modèle':
                        ''.join([
                            c for c in str(grid.best_params_['classifier']).
                            split('(')[0] if c.isupper()
                        ]) + '/' + ''.join([
                            c
                            for c in str(grid.best_params_['sampling']).split(
                                '(')[0] if c.isupper()
                        ]) + '_' + data_type,
                        'Feature':
                        test.columns,
                        'value':
                        grid.best_estimator_.named_steps['classifier'].
                        feature_importances_
                    },
                    index=pd.RangeIndex(IDX, IDX + len(test.columns)))
            ])
            IDX += len(test.columns)

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
            CMfig.write_image('./Figures/CM{}Grid{}{}.pdf'.format(
                data_type, ''.join([
                    c
                    for c in str(grid.best_params_['classifier']).split('(')[0]
                    if c.isupper()
                ]), ''.join([
                    c for c in str(grid.best_params_['sampling']).split('(')[0]
                    if c.isupper()
                ])))

    Rocfig.update_layout(xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
    Rocfig.show(renderer='notebook')
    if write_data is True:
        Rocfig.write_image(
            './Figures/CurvesROCGridBest_{}.pdf'.format(data_type))

    return Scores, feature_importance


# %%
classifier = Pipeline([('sampling', 'passthrough'),
                       ('classifier', 'passthrough')])

sampler = [
    RandomUnderSampler(random_state=0),
    RandomOverSampler(random_state=0),
    SMOTE(random_state=0, n_jobs=-1),
    SMOTETomek(random_state=0, n_jobs=-1)
]

param_grid = [{
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
    'classifier': [
        LGBMClassifier(objective='binary',
                       random_state=0,
                       device_type='gpu',
                       verbose=0)
    ],
    'classifier__boosting_type': ['gbdt', 'dart', 'rf', 'goss'],
    'classifier__n_estimators': [100, 500, 1000]
}]

Scores_base, ImpFeat_base = GridPlot(classifier,
                                     param_grid,
                                     app_train,
                                     app_test,
                                     data_type='base',
                                     sample=40000,
                                     imputerize=False)
# %%
ScoresM_base = Scores_base.melt('Modèle').rename(columns={'variable': 'Score'})
fig = px.bar(
    ScoresM_base,
    x='Score',
    y='value',
    color='Modèle',
    barmode='group',
    title='Scores pour les différents modèles<br>sans polynomial features')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresGrid_base.pdf')
# %%
PlotDF_ImpFeat_base = ImpFeat_base.sort_values(
    by='value', ascending=False).groupby('Modèle').head(10)
fig = px.bar(
    PlotDF_ImpFeat_base,
    x='value',
    y='Feature',
    orientation='h',
    labels={'value': 'Importance value'},
    facet_row='Modèle',
    facet_row_spacing=.05,
    title='Importances des variables utilisées par les différents modèles')
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None, showticklabels=True)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/BestFeatGrid_base.pdf')
# %%
Scores_FE, ImpFeat_FE = GridPlot(classifier,
                                 param_grid,
                                 trainfeat_transDF,
                                 testfeat_transDF,
                                 data_type='featEng',
                                 sample=40000,
                                 imputerize=False)
# %%
ScoresM_FE = Scores_FE.melt('Modèle').rename(columns={'variable': 'Score'})
fig = px.bar(
    ScoresM_FE,
    x='Score',
    y='value',
    color='Modèle',
    barmode='group',
    title='Scores pour les différents modèles<br>avec polynomial features')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresGrid_FE.pdf')
# %%
PlotDF_ImpFeat_FE = ImpFeat_FE.sort_values(
    by='value', ascending=False).groupby('Modèle').head(10)
fig = px.bar(
    PlotDF_ImpFeat_FE,
    x='value',
    y='Feature',
    orientation='h',
    labels={'value': 'Importance value'},
    facet_row='Modèle',
    facet_row_spacing=.05,
    title=
    'Importances des variables utilisées par les<br>différents modèles avec polynomial features'
)
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None, showticklabels=True)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/BestFeatGrid_FE.pdf')
# %%
ScoresFull = pd.concat([ScoresM_base, ScoresM_FE])
fig = px.bar(ScoresFull,
             x='Score',
             y='value',
             color='Modèle',
             barmode='group',
             title='Scores pour les différents modèles')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresGridFull.pdf')
# %% [markdown]
# L'utilisation de feature polynomiales n'améliore pas significativement nos
# résultats. Nous allons à présent tester un score plus adapté afin de prendre
# en compte le coût d'un mauvais classement
# %%
sampler = [
    RandomUnderSampler(random_state=0),
    RandomOverSampler(random_state=0)
]

param_grid = [{
    'sampling': [sampler[0]],
    'classifier': [
        XGBClassifier(tree_method='gpu_hist',
                      gpu_id=0,
                      objective='binary:logistic',
                      eval_metric='auc',
                      use_label_encoder=False,
                      random_state=0)
    ]
}, {
    'sampling': [sampler[1]],
    'classifier': [
        LGBMClassifier(objective='binary',
                       random_state=0,
                       device_type='gpu',
                       verbose=0)
    ],
    'classifier__boosting_type': ['dart'],
    'classifier__n_estimators': [100, 500, 1000]
}]
ScoresF2_base, ImpFeatF2_base = GridPlot(classifier,
                                         param_grid,
                                         app_train,
                                         app_test,
                                         data_type='base',
                                         sample=40000,
                                         refit='F2',
                                         imputerize=False)
# %%
ScoresF2M_base = ScoresF2_base.melt('Modèle').rename(
    columns={'variable': 'Score'})
fig = px.bar(ScoresF2M_base,
             x='Score',
             y='value',
             color='Modèle',
             barmode='group',
             title='Scores')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresGridF2_base.pdf')
# %%
PlotDF_ImpFeatF2_base = ImpFeatF2_base.sort_values(
    by='value', ascending=False).groupby('Modèle').head(10)
fig = px.bar(PlotDF_ImpFeatF2_base,
             x='value',
             y='Feature',
             orientation='h',
             labels={'value': 'Importance value'},
             facet_row='Modèle',
             facet_row_spacing=.05,
             title='Importances des variables utilisées')
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None, showticklabels=True)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/BestFeatGridF2_base.pdf')

# %%
if write_data is True:
    app_train[ImpFeatF2_base[(ImpFeatF2_base.value != 0) & (
        ImpFeatF2_base.Modèle == 'LGBMC/ROS_base')].Feature.to_list()].sample(
            200000, random_state=0).to_csv('TrainSet.csv')
    app_test[ImpFeatF2_base[(ImpFeatF2_base.value != 0)
                            & (ImpFeatF2_base.Modèle == 'LGBMC/ROS_base')].
             Feature.to_list()].to_csv('TestSet.csv')
# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(app_train.drop(columns='TARGET'))
train_scal = scaler.transform(app_train.drop(columns='TARGET'))

X_train, X_test, y_train, y_test = train_test_split(train_scal,
                                                    app_train.TARGET)

ROSamp = RandomOverSampler(random_state=0)
X_samp, y_samp = ROSamp.fit_resample(X_train, y_train)

model = LGBMClassifier(boosting_type='dart',
                       device_type='gpu',
                       objective='binary',
                       random_state=0,
                       n_estimators=100)
# %%
model.fit(X_samp, y_samp)
# %%
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(app_train.drop(columns='TARGET'))
# %%
shap.force_plot(explainer.expected_value[1], shap_values[1][0, :],
                app_train.drop(columns='TARGET').iloc[0, :])
# %%
shap.force_plot(explainer.expected_value[1], shap_values[1][:500, :],
                app_train.drop(columns='TARGET').iloc[:500, :])
# %%
shap.summary_plot(shap_values, app_train.drop(columns='TARGET'))

# %%
model_pred = model.predict(X_test)
model_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, ROCthresh = roc_curve(y_test, model_proba)
prec, rec, PRthresh = precision_recall_curve(y_test, model_proba)

# %%
fig = px.line(x=rec,
              y=prec,
              labels={
                  'x': 'Recall',
                  'y': 'Precision'
              },
              title='Courbe Precision-Recall')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/PRCurve.pdf')
# %%
fig = px.line(x=PRthresh,
              y=[rec[:-1], prec[:-1]],
              labels={'x': 'Threshold'},
              title='Precision et recall en fonction du threshold')
newnames = {'wide_variable_0': 'Recall', 'wide_variable_1': 'Precision'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(
                                          t.name, newnames[t.name])))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/PRThreshCurve.pdf')

# %%
fig = px.line(x=fpr,
              y=tpr,
              labels={
                  'x': 'FPR',
                  'y': 'TPR'
              },
              title='Courbe ROC')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ROCCurve.pdf')
# %%
BestPRthresh = PRthresh[np.argmax(rec - prec)]
print('Meilleur threshold pour la courbe PR : {}'.format(BestPRthresh))
# %%
BestROCthresh = ROCthresh[np.argmax(tpr - fpr)]
print('Meilleur threshold pour la courbe ROC : {}'.format(BestROCthresh))
# %%
ThreshPRPred = np.where(model_proba > BestPRthresh, 1, 0)
ThreshROCPred = np.where(model_proba > BestROCthresh, 1, 0)
# %%
CMfig = px.imshow(
    confusion_matrix(y_test, ThreshPRPred),
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
    'Matrice de confusion des prediction du modèle<br>avec un thresholdPR de {}'
    .format(round(BestPRthresh, 3)))

CMfig.update_layout(plot_bgcolor='white')
CMfig.update_coloraxes(showscale=False)
CMfig.show(renderer='notebook')

# %%
CMfig = px.imshow(
    confusion_matrix(y_test, ThreshROCPred),
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
    'Matrice de confusion des prediction du modèle<br>avec un thresholdROC de {}'
    .format(round(BestROCthresh, 3)))

CMfig.update_layout(plot_bgcolor='white')
CMfig.update_coloraxes(showscale=False)
CMfig.show(renderer='notebook')
# %%
scores = {
    'AUC': [
        roc_auc_score(y_test, model_proba),
        roc_auc_score(y_test, model_proba),
        roc_auc_score(y_test, model_proba)
    ],
    'Precision': [
        precision_score(y_test, model_pred),
        precision_score(y_test, ThreshPRPred),
        precision_score(y_test, ThreshROCPred)
    ],
    'Recall': [
        recall_score(y_test, model_pred),
        recall_score(y_test, ThreshPRPred),
        recall_score(y_test, ThreshROCPred)
    ],
    'F1': [
        f1_score(y_test, model_pred),
        f1_score(y_test, ThreshPRPred),
        f1_score(y_test, ThreshROCPred)
    ],
    'F2': [
        fbeta_score(y_test, model_pred, beta=2),
        fbeta_score(y_test, ThreshPRPred, beta=2),
        fbeta_score(y_test, ThreshROCPred, beta=2)
    ]
}

# %%
scores_thresh = pd.DataFrame.from_dict(
    scores,
    orient='index',
    columns=['thresh_05', 'threshPR_opt', 'threshROC_opt'])

# %%
