import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import utils
import dash_utils
import os
import configparser
import plotly.graph_objects as go
from scipy.stats import spearmanr
import dash_bootstrap_components as dbc
import json
from sodapy import Socrata

# Get the dataframes and combine vaccine totals
df_cases, df_deaths, df_fatality_rate, df_admin, df_second_admin = utils.update_frames()
abbrev_to_state = utils.load_pickle('data/state_abbrev.pickle')
state_to_abbrev = {b:a for a,b in abbrev_to_state.items()}
with open('data/about.txt', 'r') as file:
    text_input = file.read()#.replace('\n', '')


external_stylesheets = [dbc.themes.BOOTSTRAP] #### NEED TO FIND A PRETTY BOOTSTRAP STYLESHEET
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server
subtitle = 'Please mouseover a state to begin. Initial values are for the' \
           ' entire U.S. State-wide values are truncated to overlapping data ' \
           'points and will update automatically once a week. Please see below ' \
           'for project details.'


app.title='Covid-19 Vaccination-Mortality Correlation Data'

app.layout = html.Div([
    # App heading
    dbc.Row([
        dbc.Col(
            html.H1("U.S. Covid-19 Vaccination/Mortality Correlation", style={'textAlign':'center'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            html.H1(subtitle, style={'textAlign': 'center', 'font-size':'20px'}), width={'size':8, 'offset':2})
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='usa1_dropdown',
                options = [{'label': 'Total Vaccinations Per State', 'value':'total'},
                           {'label':'Vaccinations Administered Per Covid Fatality Per State', 'value':'relative'}],
                value= 'total'
            ),
        width=6)
    ]),
    # Mouseover U.S. Map
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='usa1'), width=6
        ),
        dbc.Col(
           dcc.Graph(id='overlay', figure={}), width=6
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='scatter_dropdown',
                options=[{'label': 'Deaths versus Vaccinations', 'value': 'unshifted'},
                         {'label': 'Deaths v. Vaccinations Shifted 3 Weeks', 'value': 'shifted'}],
                value='unshifted'
            ),
            width=6),
        dbc.Col(
            dcc.Dropdown(
                id='timeseries_dropdown',
                options=[{'label': 'Vaccinations per week', 'value': 'vaccinations'},
                         {'label': 'Deaths per week', 'value': 'deaths'}],
                value='vaccinations'
            ),
            width=6)
    ]),
    dbc.Row([
        dbc.Col(
           dcc.Graph(id='scatter'), width=6
        ),
        dbc.Col(
            dcc.Graph(id='time_fig'), width=6
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Textarea(
                id='about_text',
                value=text_input,
                style={'width':'80%', 'height':300}
            ), width={'size':8, 'offset':2})
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Button(
                'My personal website',
                id='link1',
                href='https://trenty.net/'
            ), width={'size':3, 'offset':5}
        )
    ])
    # # Code to check the mouseover output.
    # html.Div(className='row', children=[
    #     html.Div([
    #         dcc.Markdown("""
    #            **Hover Data**
    #
    #            Mouse over values in the graph.
    #        """),
    #         html.Pre(id='hover-data')
    #     ], className='three columns')])
])


# Retrieves a figure to output the geographical heatmap of the US based
# on the dropdown value.
@app.callback(Output('usa1','figure'), [Input('usa1_dropdown', 'value')])
def update_usa1(dropdown):
    if dropdown == None:
        dropdown = 'total'
    fig = dash_utils.get_usa_fig(df_admin, df_cases, dropdown)
    return fig


@app.callback(Output('overlay', 'figure'), [Input('usa1', 'hoverData')])
def update_overlay(hover):
    # Put initial value to CA.
    if hover == None:
        abbrev = 'U.S.'
        state = 'Total'
    else:
        abbrev = hover['points'][0]['location']
        state = abbrev_to_state[abbrev]
    fig = dash_utils.get_overlay_fig(df_admin, df_cases, df_deaths, state, abbrev)
    return fig


@app.callback(Output('scatter', 'figure'), [Input('usa1', 'hoverData'), Input('scatter_dropdown', 'value')])
def update_scatter(hover, dropdown):
    # Put initial value to all of US.
    if hover == None:
        abbrev = 'U.S.'
        state = 'Total'
    else:
        abbrev = hover['points'][0]['location']
        state = abbrev_to_state[abbrev]
    fig = dash_utils.get_scatter(df_admin, df_deaths, state, abbrev, shift = dropdown)
    return fig

@app.callback(Output('time_fig', 'figure'), [Input('usa1', 'hoverData'), Input('timeseries_dropdown', 'value')])
def update_time_scatter(hover, dropdown):
    # Put initial value to all of US.
    if hover == None:
        abbrev = 'U.S.'
        state = 'Total'
    else:
        abbrev = hover['points'][0]['location']
        state = abbrev_to_state[abbrev]
    if dropdown == 'vaccinations':
        df = df_admin
    else:
        df = df_deaths
    fig = dash_utils.get_time_plot(df, state, dropdown)
    return fig


# Code to check the mouseover output.
# @app.callback(
#     Output('hover-data', 'children'),
#     Input('usa1', 'hoverData'))
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)


######################
######################
######################
if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     .
#     .
#     app.run_server(host='0.0.0.0', port=8050, debug=True)