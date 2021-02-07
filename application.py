import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import utils
import os
import configparser
import plotly.graph_objects as go
from scipy.stats import spearmanr
import dash_bootstrap_components as dbc


# Get the dataframes and combine vaccine totals
formatted_list = utils.get_current_data()
df_pfizer = formatted_list[0]
df_moderna = formatted_list[1]
df_covid_deaths = formatted_list[2]
df_vaccine = utils.get_total_frame(df_pfizer, df_moderna)

external_stylesheets = [dbc.themes.BOOTSTRAP] #### NEED TO FIND A PRETTY BOOTSTRAP STYLESHEET
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server

# Call the USA fig
fig = utils.get_vacc_fig(df_vaccine)

app.title='Covid-19 Vaccination-Mortality Correlation Data'

app.layout = html.Div([
    # App heading
    dbc.Row([
        dbc.Col([
            html.H1("Mouseover a State to begin", style={'textAlign':'center'})
        ], width=12)
    ]),

    # Mouseover U.S. Map
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='usa', figure=fig), sm=6
        )
    ])
])



######################
######################
######################
if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     .
#     .
#     app.run_server(host='0.0.0.0', port=8050, debug=True)