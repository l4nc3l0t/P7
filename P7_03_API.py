from flask import Flask, jsonify, request
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier

app = Flask(__name__)

data_train = pd.read_csv('./application_train.csv')
data_test = pd.read_csv('./application_test.csv')
id_clients = data_test.SK_ID_CURR

data_train = pd.read_csv('./application_train.csv')
data_test = pd.read_csv('./application_test.csv')
id_clients = data_test.SK_ID_CURR
# encodage des colonnes catégorielles à 2 catégories
le = LabelEncoder()
le_count = 0
col_enc = []
for col in data_train.select_dtypes('object'):
    if len(data_train[col].unique()) <= 2:
        le.fit(data_train[col])
        data_train[col] = le.transform(data_train[col])
        data_test[col] = le.transform(data_test[col])
        col_enc.append(col)
        le_count += 1
# créations de "dummy variables" pour les colonnes ayant plus de 2 catégories
data_train = pd.get_dummies(data_train, dummy_na=True)
data_test = pd.get_dummies(data_test, dummy_na=True)
# création d'un flag anomalie
data_train[
    'DAYS_EMPLOYED_ANOM'] = data_train.DAYS_EMPLOYED == data_train.DAYS_EMPLOYED.max(
    )
data_train.DAYS_EMPLOYED_ANOM = data_train.DAYS_EMPLOYED_ANOM.astype(int)
data_train.DAYS_EMPLOYED.replace({data_train.DAYS_EMPLOYED.max(): np.nan},
                                 inplace=True)

data_test[
    'DAYS_EMPLOYED_ANOM'] = data_test.DAYS_EMPLOYED == data_test.DAYS_EMPLOYED.max(
    )
data_test.DAYS_EMPLOYED_ANOM = data_test.DAYS_EMPLOYED_ANOM.astype(int)
data_test.DAYS_EMPLOYED.replace({data_test.DAYS_EMPLOYED.max(): np.nan},
                                inplace=True)

# uniformisation des colonnes entre jeux train et test
data_train, data_test = data_train.align(data_test, axis=1, fill_value=np.nan)
data_test.drop(columns='TARGET', inplace=True)

# scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train.drop(columns='TARGET'))
train_scal = scaler.transform(data_train.drop(columns='TARGET'))
test_scal = scaler.transform(data_test)

X_train, X_test, y_train, y_test = train_test_split(train_scal,
                                                    data_train.TARGET)
# rééchantillonnage
ROSamp = RandomOverSampler(random_state=0)
X_samp, y_samp = ROSamp.fit_resample(X_train, y_train)

model = LGBMClassifier(boosting_type='dart',
                       device_type='gpu',
                       objective='binary',
                       random_state=0,
                       n_estimators=100)
model.fit(X_samp, y_samp)


@app.route("/ID_clients", methods=["GET"])
def ID_clients():
    return jsonify(json.loads(id_clients.to_json(orient='values')))

# http://localhost:5000/infos_client?id=100001
@app.route("/infos_client", methods=["GET"])
def show_data():
    ID_client = request.args.get("id")
    data_client = data_test[data_test.SK_ID_CURR == int(ID_client)].set_index(
        'SK_ID_CURR')
    print(data_client)
    data_reponse = json.loads(data_client.to_json(orient='index'))
    print(data_reponse)
    return jsonify(data_reponse)

# http://localhost:5000/predict?id=100001
@app.route("/predict", methods=["GET"])
def model_pred():
    ID_client = request.args.get("id")
    index = data_test[data_test.SK_ID_CURR == int(ID_client)].index
    proba = float(model.predict_proba(test_scal[index])[:,1])
    pred = int(model.predict(test_scal[index]))
    result = {
        'ID': int(data_test[data_test.SK_ID_CURR == int(ID_client)].SK_ID_CURR),
        'proba_defaut': proba,
        'prediction': pred
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="localhost", debug=True)