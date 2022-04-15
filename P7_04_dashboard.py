from dash import Dash, html, dcc, Input, Output, dash_table
import requests
import json

import pandas as pd
import numpy as np
import plotly.express as px

app = Dash(__name__)

Titre = ''' 
## Prêt à dépenser

'''

URL_API = 'http://localhost:5000/'


def get_id_list():
    ID_list = requests.get(URL_API + 'ID_clients').json()
    return ID_list


app.layout = html.Div([
    html.H4(dcc.Markdown(children='Liste des indentifiants clients :')),
    html.Div([
        'ID client :',
        dcc.Dropdown(get_id_list(), get_id_list()[0], id='ID_choosed')
    ]),
    html.Div(['Informations client :',
              dash_table.DataTable(id='table')]),
    dcc.Store(id='infos_client')
])


@app.callback(Output('infos_client', 'data'), Input('ID_choosed', 'value'))
def get_client_infos(value):
    url = URL_API + 'ID_clients/infos_client?id=' + str(value)
    data = requests.get(url).json()
    return data


@app.callback(Output('table', 'data'), Output('table', 'columns'),
              Input('infos_client', 'data'))
def table(data_client):
    df = pd.DataFrame(data_client).reset_index().rename(
        columns={'index': 'variable'})
    col = [{"name": i, "id": i} for i in df.columns]
    values = df.to_dict('records')
    print(col)
    print(values)
    return [values, col]


if __name__ == '__main__':
    app.run_server(debug=True)