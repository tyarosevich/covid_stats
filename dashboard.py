import pandas as pd
import numpy as np
from sodapy import Socrata
import utils
import re
from importlib import reload
import datetime as dt

# Import formatted data
formatted_list = utils.load_pickle('data/formatted_data.pickle')
df_pfizer_formatted = formatted_list[0]
df_moderna_formatted = formatted_list[1]
df_covid_deaths_formatted = formatted_list[2]

# Check for updates every monday
today = dt.date.today()
day_of_week = today.weekday()
updated_list = []
if day_of_week == 0:
    client = Socrata("data.cdc.gov", None)
    updated_list = utils.update_frames(formatted_list, client)

df_pfizer_updated = updated_list[0]
df_moderna_updated = updated_list[1]
df_deaths_updated = updated_list[2]
