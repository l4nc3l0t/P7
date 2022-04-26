from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import requests
import json

import pandas as pd
import numpy as np
import plotly.express as px

import shap

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

Titre = ''' 
## Prêt à dépenser

'''

URL_API = 'http://localhost:5000/'


def get_id_list():
    ID_list = requests.get(URL_API + 'ID_clients/').json()
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
        dbc.Col(dbc.Row([html.Div(id='shap_force')])),
        dbc.Row([
            dbc.Col(dbc.Row([
                html.Label('Informations client :'),
                dash_table.DataTable(id='tabledata',
                                     style_table={
                                         'height': '600px',
                                         'overflowY': 'scroll'
                                     }),
                dcc.Store(id='infos_client')
            ]),
                    width={'size': 6}),
            dbc.Col(
                dbc.Row([
                    dcc.Markdown(children='Nombre de clients similaires :'),
                    dcc.Slider(id='n_neighbors',
                               min=10,
                               max=1000,
                               step=100,
                               value=10),
                    dcc.Store(id='k_neighbors')
                ]))
        ])
    ])
])


@app.callback(Output('infos_client', 'data'), Input('ID_choosed', 'value'))
def get_client_infos(idclient):
    url = URL_API + 'ID_clients/infos_client/?id=' + str(idclient)
    data = requests.get(url).json()
    return data


@app.callback(Output('tabledata', 'data'), Output('tabledata', 'columns'),
              Input('infos_client', 'data'))
def table(data_client):
    df = pd.DataFrame(data_client).reset_index().rename(
        columns={'index': 'variable'})
    col = [{"name": i, "id": i} for i in df.columns]
    values = df.to_dict('records')
    return [values, col]


@app.callback(Output('prediction', 'data'), Input('ID_choosed', 'value'))
def get_prediction(idclient):
    url = URL_API + 'predict/?id=' + str(idclient)
    prediction = requests.get(url).json()
    return prediction


@app.callback(Output('tablepred', 'data'), Output('tablepred', 'columns'),
              Input('prediction', 'data'))
def table(prediction):
    df = pd.DataFrame(prediction, index=['a'])
    col = [{"name": i, "id": i} for i in df.columns]
    values = df.to_dict('records')
    return [values, col]


@app.callback(Output('shap_force', 'children'), Input('ID_choosed', 'value'))
def get_shap_infos(idclient):
    shap.initjs()
    url = URL_API + 'explaination/explainer/?id=' + str(idclient)
    explain = requests.get(url).json()
    url = URL_API + 'explaination/data_shap/?id=' + str(idclient)
    data_shap = requests.get(url).json()
    url = URL_API + 'explaination/data_client/?id=' + str(idclient)
    data_client = requests.get(url).json()
    print(list(data_client.keys())[0])
    df_client = pd.DataFrame(data_client[list(data_client.keys())[0]],
                             index=[0])
    print(df_client)
    force_plot = shap.force_plot(explain,
                                 np.array(data_shap[0]),
                                 df_client,
                                 matplotlib=False)
    shapF_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shapF_html,
                       style={
                           "width": "100%",
                           "height": "200px",
                           "border": 0
                       })


@app.callback(Output('k_neighbors', 'data'), Input('n_neighbors', 'value'),
              Input('ID_choosed', 'value'))
def knearestneighbors(n_neighbors, idclient):
    url = URL_API + 'neighbors/?nn=' + str(n_neighbors) + '&id=' + str(
        idclient)
    neighborslist = requests.get(url).json()
    return neighborslist


if __name__ == '__main__':
    app.run_server(debug=True)