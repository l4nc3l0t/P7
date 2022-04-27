from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import requests
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                                         'height': '900px',
                                         'overflowY': 'scroll'
                                     }),
                dcc.Store(id='infos_client')
            ]),
                    width={'size': 4}),
            dbc.Col(
                dbc.Row([
                    dcc.Markdown(children='Nombre de clients similaires :'),
                    dcc.Slider(id='n_neighbors',
                               min=10,
                               max=1000,
                               step=100,
                               value=100),
                    dcc.Store(id='neighbors_data'),
                    dcc.Tabs(children=[
                        dcc.Tab(label='Comparaison EXT_SOURCES',
                                children=[
                                    dcc.Graph(id='compareEXT_neighbors'),
                                    dcc.Graph(id='compareEXT_full')
                                ]),
                        dcc.Tab(
                            label='Comparaison temps employé/enregistrement',
                            children=[
                                dcc.Graph(id='compareDAYS_neighbors'),
                                dcc.Graph(id='compareDAYS_full')
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
    Output('compareEXT_neighbors', 'figure'),
    Input('neighbors_data', 'data'),
    Input('infos_client', 'data'),
)
def neighbors_compare(neighbors_data, client_data):
    client_data = pd.DataFrame(client_data).T

    neighbors_data = pd.DataFrame(neighbors_data).T
    ndD = neighbors_data[neighbors_data.TARGET == 1]
    ndND = neighbors_data[neighbors_data.TARGET == 0]

    col_interest = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    cd = client_data[col_interest]

    ndDMean = ndD[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})
    ndNDMean = ndND[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{
            'type': 'polar'
        }] * 2],
        subplot_titles=('Moyenne des clients similaires ne faisant pas défaut',
                        'Moyenne des clients similaires faisant défaut'))
    fig.add_trace(
        go.Scatterpolar(
            r=ndNDMean[0],
            theta=ndNDMean.variables,
            mode='lines',
            name='Moyenne clients similaires ne faisant pas défaut'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=ndDMean[0],
                        theta=ndDMean.variables,
                        mode='lines',
                        name='Moyenne clients similaires faisant défaut'), 1,
        2)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 2)

    fig.update_layout(height=400)
    fig.update_annotations(yshift=10)
    return fig


@app.callback(Output('compareEXT_full', 'figure'),
              Input('infos_client', 'data'))
def full_compare(client_data):
    url = URL_API + 'full_data/'
    full_dataJ = requests.get(url).json()
    full_data = pd.DataFrame(full_dataJ).T
    print(full_data)
    fdD = full_data[full_data.TARGET == 1]
    fdND = full_data[full_data.TARGET == 0]

    client_data = pd.DataFrame(client_data).T

    col_interest = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    cd = client_data[col_interest]

    fdDMean = fdD[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})
    fdNDMean = fdND[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{
            'type': 'polar'
        }] * 2],
        subplot_titles=("Moyenne de l'ensemble des clients ne faisant défaut",
                        "Moyenne de l'ensemble des client faisant défaut"))

    fig.add_trace(
        go.Scatterpolar(r=fdNDMean[0],
                        theta=fdNDMean.variables,
                        mode='lines',
                        name='Moyenne clients ne faisant pas défaut'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=fdDMean[0],
                        theta=fdDMean.variables,
                        mode='lines',
                        name='Moyenne clients faisant défaut'), 1, 2)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 2)

    fig.update_layout(height=400)
    fig.update_annotations(yshift=10)
    return fig


@app.callback(
    Output('compareDAYS_neighbors', 'figure'),
    Input('neighbors_data', 'data'),
    Input('infos_client', 'data'),
)
def neighbors_compare(neighbors_data, client_data):
    client_data = pd.DataFrame(client_data).T

    neighbors_data = pd.DataFrame(neighbors_data).T
    neighbors_data.DAYS_EMPLOYED = neighbors_data.DAYS_EMPLOYED / -365
    neighbors_data.DAYS_BIRTH = neighbors_data.DAYS_BIRTH / -365
    neighbors_data.DAYS_REGISTRATION = neighbors_data.DAYS_REGISTRATION / -365

    ndD = neighbors_data[neighbors_data.TARGET == 1]
    ndND = neighbors_data[neighbors_data.TARGET == 0]

    col_interest = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION']

    cd = client_data[col_interest]
    cd.DAYS_EMPLOYED = cd.DAYS_EMPLOYED / -365
    cd.DAYS_BIRTH = cd.DAYS_BIRTH / -365
    cd.DAYS_REGISTRATION = cd.DAYS_REGISTRATION / -365

    ndDMean = ndD[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})
    ndNDMean = ndND[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{
            'type': 'polar'
        }] * 2],
        subplot_titles=('Moyenne des clients similaires ne faisant pas défaut',
                        'Moyenne des clients similaires faisant défaut'))
    fig.add_trace(
        go.Scatterpolar(
            r=ndNDMean[0],
            theta=ndNDMean.variables,
            mode='lines',
            name='Moyenne clients similaires ne faisant pas défaut'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=ndDMean[0],
                        theta=ndDMean.variables,
                        mode='lines',
                        name='Moyenne clients similaires faisant défaut'), 1,
        2)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 2)

    fig.update_layout(height=400)
    fig.update_annotations(yshift=10)
    return fig


@app.callback(Output('compareDAYS_full', 'figure'),
              Input('infos_client', 'data'))
def full_compare(client_data):
    url = URL_API + 'full_data/'
    full_dataJ = requests.get(url).json()
    full_data = pd.DataFrame(full_dataJ).T
    full_data.DAYS_EMPLOYED = full_data.DAYS_EMPLOYED / -365
    full_data.DAYS_BIRTH = full_data.DAYS_BIRTH / -365
    full_data.DAYS_REGISTRATION = full_data.DAYS_REGISTRATION / -365

    print(full_data)
    fdD = full_data[full_data.TARGET == 1]
    fdND = full_data[full_data.TARGET == 0]

    client_data = pd.DataFrame(client_data).T

    col_interest = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION']

    cd = client_data[col_interest]
    cd.DAYS_EMPLOYED = cd.DAYS_EMPLOYED / -365
    cd.DAYS_BIRTH = cd.DAYS_BIRTH / -365
    cd.DAYS_REGISTRATION = cd.DAYS_REGISTRATION / -365

    fdDMean = fdD[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})
    fdNDMean = fdND[col_interest].mean().reset_index().rename(
        columns={'index': 'variables'})

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{
            'type': 'polar'
        }] * 2],
        subplot_titles=("Moyenne de l'ensemble des clients ne faisant défaut",
                        "Moyenne de l'ensemble des client faisant défaut"))

    fig.add_trace(
        go.Scatterpolar(r=fdNDMean[0],
                        theta=fdNDMean.variables,
                        mode='lines',
                        name='Moyenne clients ne faisant pas défaut'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 1)
    fig.add_trace(
        go.Scatterpolar(r=fdDMean[0],
                        theta=fdDMean.variables,
                        mode='lines',
                        name='Moyenne clients faisant défaut'), 1, 2)
    fig.add_trace(
        go.Scatterpolar(r=cd.iloc[0],
                        theta=cd.columns,
                        mode='lines',
                        name='Données client'), 1, 2)

    fig.update_layout(height=400)
    fig.update_annotations(yshift=10)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)