import pandas as pd
import numpy as np
from sodapy import Socrata
import utils
import re
from importlib import reload
import datetime as dt
#%%

try:
    unprocessed_data = utils.load_pickle('data/unprocessed_data.pickle')
    df_pfizer = unprocessed_data[0]
    df_moderna = unprocessed_data[1]
    df_covid_deaths = unprocessed_data[2]

except FileNotFoundError:
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    client = Socrata("data.cdc.gov", None)

    # Get jsons from CDC
    results_pfizer = client.get("saz5-9hgg", limit=1000)
    results_moderna = client.get("b7pe-5nws", limit=1000)
    covid_deaths = client.get("muzy-jte6", limit=4000)

    # Convert to pandas DataFrame
    df_pfizer = pd.DataFrame.from_records(results_pfizer)
    df_moderna = pd.DataFrame.from_records(results_moderna)
    df_covid_deaths = pd.DataFrame.from_records(covid_deaths)
    unprocessed_data = [df_pfizer, df_moderna, df_covid_deaths]
    utils.save_pickle('data/unprocessed_data.pickle', unprocessed_data)

try:
    formatted_list = utils.load_pickle('data/formatted_data.pickle')
    df_pfizer_formatted = formatted_list[0]
    df_moderna_formatted = formatted_list[1]
    df_covid_deaths_formatted = formatted_list[2]

except FileNotFoundError:



    # Reformat to needed columns, and column labels that are datetime ready
    # strings. Column labels are all moved to 'week ending on' the date, which is
    # a saturday, to conform to other CDC data.
    df_pfizer_formatted = utils.clean_frame(df_pfizer, 'pfizer')
    df_moderna_formatted = utils.clean_frame(df_moderna, 'moderna')

    # Totally clean, re-format and re-aggregate this disaster of a dataset. Pfizer frame
    # is passed as template.
    df_covid_deaths_formatted = utils.correct_bad_aggreg(utils.clean_frame(df_covid_deaths, 'covid_deaths'), df_pfizer_formatted)
    # Save formatted data
    utils.save_pickle('data/formatted_data.pickle', [df_pfizer_formatted, df_moderna_formatted, df_covid_deaths_formatted])



