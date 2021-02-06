import pandas as pd
import numpy as np
from sodapy import Socrata
import utils
import re
from importlib import reload
import datetime as dt

# Import formatted data
formatted_list = utils.get_current_data()

df_pfizer_formatted = formatted_list[0]
df_moderna_formatted = formatted_list[1]
df_covid_deaths_formatted = formatted_list[2]



#%%

df_vaccine = utils.get_total_frame(df_pfizer_formatted, df_moderna_formatted)
