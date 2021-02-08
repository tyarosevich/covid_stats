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
import json


# Get the dataframes and combine vaccine totals
formatted_list = utils.get_current_data()
df_pfizer = formatted_list[0]
df_moderna = formatted_list[1]
df_covid_deaths = formatted_list[2]
df_vaccine = utils.get_total_frame(df_pfizer, df_moderna)
df_population = utils.load_pickle('data/df_population.pickle')
df_vacc_withpop_deaths = utils.get_df_pop_and_deaths(df_vaccine, df_population, df_covid_deaths)

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
                           {'label':'Vaccinations Per 1,000 Residents', 'value':'relative'}],
                value= 'total'
            ),
        width=6)
    ]),
    # Mouseover U.S. Map
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='usa1')# , sm=5
        ),
        dbc.Col(
           dcc.Graph(id='scatter')#, sm=5
        )
    ])
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
    fig = utils.get_vacc_fig(df_vacc_withpop_deaths, dropdown)
    return fig

@app.callback(Output('scatter', 'figure'), Input('usa1', 'hoverData'))
def update_scatter(hover):
    # Put initial value to CA.
    if hover == None:
        state = 'CA'
    else:
        state = hover['points'][0]['location']
    fig = utils.get_scatter(df_vaccine, df_covid_deaths, state)
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