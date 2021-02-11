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
df_pfizer, df_moderna, df_cases, df_deaths, df_fatality_rate, df_admin, df_second_admin = utils.update_frames()
abbrev_to_state = utils.load_pickle('data/state_abbrev.pickle')
state_to_abbrev = {b:a for a,b in abbrev_to_state.items()}


external_stylesheets = [dbc.themes.BOOTSTRAP] #### NEED TO FIND A PRETTY BOOTSTRAP STYLESHEET
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server


# # Call the USA fig
# fig = utils.get_vacc_fig(df_vaccine)

app.title='Covid-19 Vaccination-Mortality Correlation Data'

app.layout = html.Div([
    # App heading
    dbc.Row([
        dbc.Col(
            html.H1("Mouseover a State to begin", style={'textAlign':'center'}), width=12)
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
           dcc.Graph(id='scatter'), width=6
        ),
        dbc.Col(
                html.H2(id='pearson'),
                style={'width': '30%', 'display': 'inline-block', 'text-align': 'center',
                       'vertical-align': 'middle'}, width=6
        )
    ], align='center')
    # Code to check the mouseover output.
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
@app.callback(Output('usa1','figure'), Input('usa1_dropdown', 'value'))
def update_usa1(dropdown):
    if dropdown == None:
        dropdown = 'total'
    fig = dash_utils.get_usa_fig(df_admin, df_cases, dropdown)
    return fig


@app.callback(Output('scatter', 'figure'), Input('usa1', 'hoverData'))
def update_scatter(hover):
    # Put initial value to CA.
    if hover == None:
        abbrev = 'U.S.'
        state = 'Total'
    else:
        abbrev = hover['points'][0]['location']
        state = abbrev_to_state[abbrev]
    fig = dash_utils.get_scatter(df_admin, df_fatality_rate, state, abbrev)
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

@app.callback(Output('pearson', 'children'), Input('usa1', 'hoverData'))
def update_pearsons(hover):
    if hover == None:
        abbrev = 'U.S.'
        state = 'Total'
    else:
        abbrev = hover['points'][0]['location']
        state = abbrev_to_state[abbrev]
    output = dash_utils.get_pearson(df_admin, df_fatality_rate, state, abbrev)

    return output

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