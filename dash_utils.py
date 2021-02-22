import pickle
import re
import datetime as dt
import pandas as pd
from sodapy import Socrata
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import os
import configparser
import plotly.graph_objects as go
from scipy.stats import spearmanr, linregress, pearsonr
import dash_bootstrap_components as dbc



def get_scatter(df1, df2, state, abbrev, shift='unshifted'):
    '''
    Creates a figure plotting a particular state's vaccination running total
    against its fatality rate to try to show the expected relationship.
    :param df1: DataFrame
    The vaccination dataframe.
    :param df2: DataFrame
    The deaths dataframe.
    :param state: str
    The state input from hoverdata.
    :param abbrev: str
    Abbreviation for the title.
    :return:
    '''
    if shift == 'shifted':
        # This shifts the vaccination data forward 3 weeks, which roughly
        # corresponds to when immunization would impact fatality rates (18-19 days
        # on average)
        x_arr = df1.loc[state].iloc[1:-1].to_numpy().astype(float)[0:-3]
        y_arr = df2.loc[state].iloc[1:-1].to_numpy().astype(float)[3:]
        pearson_obj = pearsonr(x_arr, y_arr)
        title = ['Vaccinations shifted up 3 weeks vs Covid-19 Fatality Rate in {}'.format(abbrev),
                'The pearson correlation is r={}'.format(round(pearson_obj[0], 4))]
        title = '<br>'.join(title)
    else:
        x_arr = df1.loc[state].iloc[1:-1].to_numpy().astype(float)
        y_arr = df2.loc[state].iloc[1:-1].to_numpy().astype(float)
        pearson_obj = pearsonr(x_arr, y_arr)
        title = ['Vaccinations Administered vs Covid-19 Fatality Rate in {}'.format(abbrev),
                'The pearson correlation is r={}'.format(round(pearson_obj[0], 4))]
        title = '<br>'.join(title)
    r = round(pearson_obj[0], 4)
    regress_obj = linregress(x_arr, y_arr)
    y_regr = regress_obj.intercept + regress_obj.slope * x_arr
    fig = go.Figure(data=go.Scattergl(
        x=x_arr,
        y=y_arr,
        mode='markers',
        marker=dict(
            color=np.random.randn(1000),
            colorscale='Blues',
            line_width=1,
            size=8
        )
    ))
    fig.add_trace(go.Scatter(
        x=x_arr,
        y=y_regr,
        mode='lines',
        line_color='#6CA6CD',
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F0F8FF',
        showlegend=False,
        xaxis_title='Cumulative Vaccine Doses Administered',
        yaxis_title='Weekly Deaths cause by Covid-19',
    )

    return fig


def get_usa_fig(df1, df2, dropdown):
    if dropdown == 'total':
        col=df1['Total'].iloc[0:-1]
        colorbar = 'Vaccine Doses Administered'
        title = 'Vaccine Doses Administered by State'
    else:
        col=np.log( df1['Total'].iloc[0:-1] / (df2['Total'].iloc[0:-1] + .0001) )
        colorbar='Vaccines Administered Per Covid Case (Log scale)'
        title = 'Vaccine Doses Administered per Case by State (Log scale)'
    # max = np.sort(df['relative'])[-4]
    # Arbitrarily setting these states to the fourth highest * 2 so the
    # so that the heat map is intelligible
    # df.loc['North Carolina', 'relative'] = max
    # df.loc['Alaska', 'relative'] = max
    # df.loc['Hawaii', 'relative'] = max
    # df['log_relative'] = np.log(df['relative'])

    fig = go.Figure(data=go.Choropleth(
        locations=df1['abbrev'],  # Column with two-letter state abbrevs.
        z=col.iloc[0:-1].astype(float),  # State vaccine totals.
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Blues',
        colorbar_title=colorbar,
    ))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        geo_scope='usa',  # limit map scope to USA
    )
    return fig

def get_overlay_fig(df1, df2, df3, state, abbrev):
    '''
    Returns a log scale overlay of the vaccine running total, cases, deaths.
    :param df1: DataFrame
    The vaccination frame.
    :param df2: DataFrame
    The cases frame.
    :param df3: DataFrame
    The deaths frame
    :param state: str
    The state (or United States) in question.
    :param abbrev: str
    State abbreviation
    :return: Figure
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df1.columns.values), y=np.log(df1.loc[state].iloc[1:-1].astype(float)),
        fillcolor='#B0E2FF',
        line_color='#FFFFFF',
        fill='tozeroy',
        name='Vaccinations'
    ))
    fig.add_trace(go.Scatter(
        x=list(df1.columns.values), y=np.log(df2.loc[state].iloc[1:-1].astype(float)),
        fillcolor='#87CEFA',
        line_color='#87CEEB',
        fill='tozeroy',
        name='Cases'
    ))
    fig.add_trace(go.Scatter(
        x=list(df1.columns.values), y=np.log(df3.loc[state].iloc[1:-1].astype(float)),
        fillcolor='#6CA6CD',
        line_color='#6495ED',
        fill='tozeroy',
        name='Deaths'
    ))

    fig.update_layout(
        title='Vaccines Administered / Cases / Deaths Overlay for {} (Log Scale)'.format(abbrev),
        title_x=0.5,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F0F8FF',
        xaxis_title='Weekly values are normalized to the next Saturday',
    )

    return fig

def get_full_scatter(df1, df2, state_dict):
    '''
    Returns a px.scatter figure for all vaccinations
    and covid deaths in the available data (all states).
    :param df1: DataFrame
    The vaccination frame.
    :param df2: DataFrame
    The covid deaths frame.
    :return: figure
    '''
    # Drop totals, stack rows into a series, and make a dataframe for plotting.
    df_prep = df1.drop(['abbrev', 'Total'], axis=1)
    df_prep.drop('Total', axis=0, inplace=True)
    x_list = df_prep.stack(dropna=False)
    df_prep2 = df2.drop(['abbrev', 'Total'], axis=1)
    df_prep2.drop('Total', axis=0, inplace=True)
    y_list = df_prep2.stack(dropna=False)
    s_state = x_list.index.to_list()
    state_list = [tup[0] for tup in s_state]
    df_full_scatter = pd.DataFrame([x_list, y_list]).transpose().rename(columns={0:'x', 1:'y'})
    df_full_scatter['label'] = state_list
    state_list = list(state_dict.keys())
    color_vals = np.arange(len(state_list))
    color_dict = dict(zip(state_list, color_vals))
    df_full_scatter['colors'] = df_full_scatter['label'].map(color_dict)

    fig = go.Figure(data=go.Scattergl(
        x=df_full_scatter['x'],
        y=df_full_scatter['y'],
        mode='markers',
        text=df_full_scatter['label'],
        marker=dict(
            color=df_full_scatter['colors'],
            colorscale='Blues',
            line_width=1
        )
    ))
    fig.update_layout(
        title='Cumulative data for all states and weeks (highlight sections to examine)',
        title_x=0.5,
        xaxis_title='Doses Administered',
        yaxis_title='Known deaths caused by Covid-19',
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F0F8FF'
    )
    return fig
