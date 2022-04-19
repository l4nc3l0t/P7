from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import requests
import json

import pandas as pd
import numpy as np
import plotly.express as px

import shap
shap.initjs()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

Titre = ''' 
## Prêt à dépenser

'''

URL_API = 'http://localhost:5000/'


def get_id_list():
    ID_list = requests.get(URL_API + 'ID_clients').json()
    return ID_list


app.layout = html.Div([
    dbc.Row([
        dbc.Col(dbc.Row([
            dcc.Markdown(children='Liste des indentifiants clients :'),
            dcc.Dropdown(get_id_list(),
                         get_id_list()[0],
                         id='ID_choosed',
                         style={"width": "200px"}),
            dcc.Markdown(children='Données de classification prédites :'),
            dash_table.DataTable(id='tablepred'),
            dcc.Store(id='prediction')
        ]),
                width={'size': 4}),
        dbc.Col(
            dbc.Row([
                html.Label('Informations client :'),
                dash_table.DataTable(id='tabledata',
                                     style_table={
                                         'height': '600px',
                                         'overflowY': 'scroll'
                                     }),
                dcc.Store(id='infos_client')
            ],
                    justify="center"))
    ])
])


@app.callback(Output('infos_client', 'data'), Input('ID_choosed', 'value'))
def get_client_infos(idclient):
    url = URL_API + 'ID_clients/infos_client?id=' + str(idclient)
    data = requests.get(url).json()
    return data


@app.callback(Output('tabledata', 'data'), Output('tabledata', 'columns'),
              Input('infos_client', 'data'))
def table(data_client):
    df = pd.DataFrame(data_client).reset_index().rename(
        columns={'index': 'variable'})
    col = [{"name": i, "id": i} for i in df.columns]
    values = df.to_dict('records')
    print(col)
    print(values)
    return [values, col]


@app.callback(Output('prediction', 'data'), Input('ID_choosed', 'value'))
def get_prediction(idclient):
    url = URL_API + 'predict?id=' + str(idclient)
    prediction = requests.get(url).json()
    return prediction


@app.callback(Output('tablepred', 'data'), Output('tablepred', 'columns'),
              Input('prediction', 'data'))
def table(prediction):
    df = pd.DataFrame(prediction,
                      index=['a'])
    col = [{"name": i, "id": i} for i in df.columns]
    values = df.to_dict('records')
    print(col)
    print(values)
    return [values, col]


if __name__ == '__main__':
    app.run_server(debug=True)