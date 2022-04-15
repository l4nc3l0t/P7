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
    html.Div(
        ['Informations client :',
         dash_table.DataTable(id='infos_client')])
])


@app.callback(Output('infos_client', 'data'), Output('infos_client', 'col'),
              Input('ID_choosed', 'value'))
def get_client_infos(value):
    data = requests.get(URL_API + 'ID_clients/infos_client?=' +
                        str(value)).json()
    print(data)
    df = pd.DataFrame(data).reset_index().rename(columns={'index':'variable'})
    col = df.columns
    values = df.to_dict('records')
    print(col)
    print(values)
    return [values, col]


if __name__ == '__main__':
    app.run_server(debug=True)