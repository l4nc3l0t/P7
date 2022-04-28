from dash import Dash, Input, Output, dash_table, dcc, html
import dash_bootstrap_components as dbc

import requests
import json

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import shap

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

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
                                         'height': '900px',
                                         'overflowY': 'scroll'
                                     }),
                dcc.Store(id='infos_client')
            ]),
                    width={'size': 4}),
            dbc.Col(
                dbc.Row([
                    dcc.Tabs(children=[
                        dcc.Tab(
                            label='Comparaison avec des clients similaires',
                            children=[
                                dcc.Markdown(
                                    children='Nombre de clients similaires :'),
                                dcc.Slider(id='n_neighbors',
                                           min=10,
                                           max=1000,
                                           step=100,
                                           value=100),
                                dcc.Store(id='neighbors_data'),
                                dcc.Graph(id='compare_neighbors',
                                          style={'height': 800}),
                            ]),
                        dcc.Tab(
                            label="Comparaison avec l'ensemble des clients",
                            children=[
                                dcc.Graph(id='compare_full',
                                          style={'height': 800})
                            ])
                    ])
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
    df_client = pd.DataFrame(data_client[list(data_client.keys())[0]],
                             index=[0])
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


@app.callback(Output('neighbors_data', 'data'), Input('n_neighbors', 'value'),
              Input('ID_choosed', 'value'))
def knearestneighbors(n_neighbors, idclient):
    url = URL_API + 'neighbors/?nn=' + str(n_neighbors) + '&id=' + str(
        idclient)
    neighborsdata = requests.get(url).json()
    return neighborsdata


@app.callback(
    Output('compare_neighbors', 'figure'),
    Input('neighbors_data', 'data'),
    Input('infos_client', 'data'),
)
def neighbors_compare(neighbors_data, client_data):
    client_data = pd.DataFrame(client_data).T

    neighbors_data = pd.DataFrame(neighbors_data).T
    ndD = neighbors_data[neighbors_data.TARGET == 1]
    ndND = neighbors_data[neighbors_data.TARGET == 0]

    col_EXT = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    col_DAYS = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION']
    col_AMT = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[names for names in (*col_EXT, *col_DAYS, *col_AMT)])
    for EXT, colEXT in enumerate(col_EXT):
        fig.add_scatter(y=client_data[colEXT],
                        mode='markers',
                        name='Client',
                        row=1,
                        col=EXT + 1)
        fig.add_trace(
            go.Box(y=ndD[colEXT],
                   boxmean='sd',
                   name='Clients similaires<br>faisant défaut'), 1, EXT + 1)
        fig.add_trace(
            go.Box(y=ndND[colEXT],
                   boxmean='sd',
                   name='Clients similaires<br>ne faisant pas défaut'), 1,
            EXT + 1)
    for DAYS, colDAYS in enumerate(col_DAYS):
        fig.add_scatter(y=client_data[colDAYS] / -365,
                        mode='markers',
                        name='Client',
                        row=2,
                        col=DAYS + 1)
        fig.add_trace(
            go.Box(y=ndD[colDAYS] / -365,
                   boxmean='sd',
                   name='Clients similaires<br>faisant défaut'), 2, DAYS + 1)
        fig.add_trace(
            go.Box(y=ndND[colDAYS] / -365,
                   boxmean='sd',
                   name='Clients similaires<br>ne faisant pas défaut'), 2,
            DAYS + 1)
    for AMT, colAMT in enumerate(col_AMT):
        fig.add_scatter(y=client_data[colAMT],
                        mode='markers',
                        name='Client',
                        row=3,
                        col=AMT + 1)
        fig.add_trace(
            go.Box(y=ndD[colAMT],
                   boxmean='sd',
                   name='Clients similaires<br>faisant défaut'), 3, AMT + 1)
        fig.add_trace(
            go.Box(y=ndND[colAMT],
                   boxmean='sd',
                   name='Clients similaires<br>ne faisant pas défaut'), 3,
            AMT + 1)
    fig.update_layout(height=800, showlegend=False)
    return fig


@app.callback(Output('compare_full', 'figure'), Input('infos_client', 'data'))
def full_compare(client_data):
    url = URL_API + 'full_data/'
    full_dataJ = requests.get(url).json()
    full_data = pd.DataFrame(full_dataJ).T
    print(full_data)
    fdD = full_data[full_data.TARGET == 1]
    fdND = full_data[full_data.TARGET == 0]

    client_data = pd.DataFrame(client_data).T

    col_EXT = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    col_DAYS = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION']
    col_AMT = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[names for names in (*col_EXT, *col_DAYS, *col_AMT)])
    for EXT, colEXT in enumerate(col_EXT):
        fig.add_scatter(y=client_data[colEXT],
                        mode='markers',
                        name='Client',
                        row=1,
                        col=EXT + 1)
        fig.add_trace(
            go.Box(y=fdD[colEXT],
                   boxmean='sd',
                   name='Clients similaires<br>faisant défaut'), 1, EXT + 1)
        fig.add_trace(
            go.Box(y=fdND[colEXT],
                   boxmean='sd',
                   name='Clients similaires<br>ne faisant pas défaut'), 1,
            EXT + 1)
    for DAYS, colDAYS in enumerate(col_DAYS):
        fig.add_scatter(y=client_data[colDAYS] / -365,
                        mode='markers',
                        name='Client',
                        row=2,
                        col=DAYS + 1)
        fig.add_trace(
            go.Box(y=fdD[colDAYS] / -365,
                   boxmean='sd',
                   name='Clients similaires<br>faisant défaut'), 2, DAYS + 1)
        fig.add_trace(
            go.Box(y=fdND[colDAYS] / -365,
                   boxmean='sd',
                   name='Clients similaires<br>ne faisant pas défaut'), 2,
            DAYS + 1)
    for AMT, colAMT in enumerate(col_AMT):
        fig.add_scatter(y=client_data[colAMT],
                        mode='markers',
                        name='Client',
                        row=3,
                        col=AMT + 1)
        fig.add_trace(
            go.Box(y=fdD[colAMT],
                   boxmean='sd',
                   name='Clients similaires<br>faisant défaut'), 3, AMT + 1)
        fig.add_trace(
            go.Box(y=fdND[colAMT],
                   boxmean='sd',
                   name='Clients similaires<br>ne faisant pas défaut'), 3,
            AMT + 1)
    fig.update_layout(height=800, showlegend=False)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
