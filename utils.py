import pickle
import re
import datetime as dt
import pandas as pd
from sodapy import Socrata
import numpy as np

def load_pickle(path):
    '''
    Loads a file
    Parameters
    ----------
    path: str
        local or full path of file

    Returns
    -------
    '''
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

def save_pickle(path, item):
    '''
    Pickles a file
    Parameters
    ----------
    path: str
    item: Any

    Returns
    -------

    '''
    with open(path, 'wb') as f:
        pickle.dump(item, f)

def fix_date_label(input_string, stamp = False, shift = 0):
    '''
    Converts the CDC API output labels to timestamps
    :param input_string: str
        The column head from the API's output.
    :param stamp: bool
        Boolean to return a datetime object or a string.
    :param shift: int
        Amount (days) to shift date by if so desired.
    :return: str
        Timestamp string 'YYYY-MM-DD'
    '''
    input_string = re.sub('^\\D*', '', input_string)
    month_day = input_string.split('_')
    if int(month_day[0]) == 12:
        timestamp = '2020-12-{}'.format(month_day[1])
    else:
        timestamp = '2021-{}-{}'.format(month_day[0], month_day[1])

    try:
        timestamp = dt.date.fromisoformat(timestamp)
    except ValueError:
        print('Source date-time label/format/standards have changed.')
    if shift > 0:
        timestamp += dt.timedelta(days=shift)
    if stamp:
        return timestamp
    else:
        return str(timestamp)

def clean_frame(df, type):
    '''
    Reformats the dataframe for the specific purposes of this app. Keeps state
    and vaccine distribution by week (as week ending on day).
    :param df: DataFrame
        Input dataframe.
    :param vacc: str
        Moderna or Pfizer.
    :return: df
        Reformatted dataframe.
    '''
    state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut",
                   "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois",
                   "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan",
                   "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska",
                   "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon",
                   "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
                   "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    abbrev_to_state = load_pickle('data/state_abbrev.pickle')
    state_to_abbrev = {b: a for a, b in abbrev_to_state.items()}

    columns = list(df.columns.values)
    if type == 'pfizer':
        columns_keep = ['jurisdiction', 'first_doses_12_14'] + [x for x in columns if x[0:8] == 'doses_al' or x[0:8] == 'doses_di']
    elif type == 'moderna':
        columns_keep = ['jurisdiction'] + [x for x in columns if x[0:8] == 'doses_al' or x[0:8] == 'doses_di']
    try:
        df = df[columns_keep]
    except KeyError:
        print("Source columns have changed.")
    columns = list(df.columns.values)[1:]
    column_labels_asdates = [fix_date_label(x, stamp=True, shift=0) for x in columns]
    column_labels_normalized = [str(normalize_day(x)) for x in column_labels_asdates]
    rename_dict = dict(zip(columns, column_labels_normalized))
    rename_dict['jurisdiction'] = 'state'
    df = df.rename(rename_dict, axis=1)

    # Removes trailing characters from CDC caveats.
    df['state'] = [x.rstrip('*').rstrip('~') for x in df['state']]
    df.reset_index(inplace=True, drop=True)
    df.fillna(0, inplace=True)
    columns = list(df.columns.values)[1:]

    # Convert string numericals to ints and N/A to 0.
    columns = list(df.columns.values)[1:]
    for col in columns:
        df[col] = df[col].apply(str_to_int)
    # Set the index to the state.
    df.set_index('state', drop=False, inplace=True)
    df = df.loc[state_names, :]
    df.rename(columns={df.columns[0]: "abbrev"}, inplace=True)
    df['abbrev'] = df['abbrev'].apply(lambda x: state_to_abbrev[x])
    return df

def normalize_day(date):
    '''
    Moves the timestamp up to the next saturday.
    :param date: date
    :return: date
    '''
    if date.weekday() == 6:
        date = date + dt.timedelta(days=6)
    else:
        date = date + dt.timedelta(days=5-date.weekday())
    return date

# Dumb little fun to replace string numbers with commas in the pfizer
# and moderna datasets.
def str_to_int(str):
    if str=='N/A':
        return 0
    else:
        return int(str.replace(',', ''))


def update_frames(force_save=False):

    # Old attempt at dynamic update.
    # df_pfizer = list_frames[0]
    # df_moderna = list_frames[1]
    # df_deaths = list_frames[2]
    # df_pfizer_new = pfiz_mod_updates(df_pfizer, client)
    # df_moderna_new = pfiz_mod_updates(df_moderna, client)
    # df_deaths_new = covid_deaths_updates(df_deaths, df_pfizer, client)
    #
    # return [df_pfizer_new, df_moderna_new, df_deaths_new]

    # Easier to just grab JSONs again, they are small.

    # Get the day of the week and load the update completed flag.
    # (this is a hacky work around to using a lib for such a simple task).
    flag = load_pickle('data/flag.pickle')
    today = dt.date.today()
    day_of_week = today.weekday()
    # If it's monday, run update and flag the update as done.
    if day_of_week == 0 and flag == 0 or force_save:
        flag = 1
        save_pickle('data/flag.pickle', flag)
        client = Socrata("data.cdc.gov", None)
        # Get jsons from CDC
        results_pfizer = client.get("saz5-9hgg", limit=1000)
        results_moderna = client.get("b7pe-5nws", limit=1000)
        # covid_deaths = client.get("muzy-jte6", limit=6000)

        # Convert to pandas DataFrame
        df_pfizer = pd.DataFrame.from_records(results_pfizer)
        df_moderna = pd.DataFrame.from_records(results_moderna)
        # df_covid_deaths = pd.DataFrame.from_records(covid_deaths)

        # Collects all the data from APIs/Github, normalizes them to the largest
        # shared span of time, and adds total rows and columns.
        df_pfizer_formatted = clean_frame(df_pfizer, 'pfizer')
        df_moderna_formatted = clean_frame(df_moderna, 'moderna')
        df_cases, df_deaths = get_case_deaths(df_pfizer_formatted)
        df_fatality_rate = get_fatality_rate(df_deaths, df_cases)
        df_admin, df_second_admin = get_administered(df_pfizer_formatted)
        frame_list = normalize_frames([df_pfizer_formatted, df_moderna_formatted,
                                       df_cases, df_deaths, df_fatality_rate, df_admin, df_second_admin])
        frame_list = add_totals(frame_list)
        # Save formatted data
        save_pickle('data/formatted_data.pickle', frame_list)

        return frame_list
    # If it's not monday, flag for update.
    else:
        flag = 0
        save_pickle('data/flag.pickle', flag)
        frame_list = load_pickle('data/formatted_data.pickle')
        return frame_list


def total_rows_columns(df):
    # Total sum per column:
    df.loc['Total', 1:] = df.iloc[:, 1:].sum(axis=0)

    # Total sum per row:
    df.loc[:, 'Total'] = df.iloc[:, 1:].sum(axis=1)

    return df


def get_case_deaths(df_format):
    '''
    Accesses the CDC API for Covid-19 cases and deaths and returns dataframes
    with more appropriate format.
    :param df_format: DataFrame
    The frame to format against, assumed produced by clean_frames above.
    :return: list
    '''

    # Load abbreviation to state dicts.
    abbrev_to_state = load_pickle('data/state_abbrev.pickle')
    state_to_abbrev = {b: a for a, b in abbrev_to_state.items()}

    # Socrata client.
    client = Socrata("data.cdc.gov", None)

    # Pulls the data from the API filtering columns and dates.
    query = 'submission_date > "2020-12-01T00:00:00.000"'
    columns = 'submission_date, state, tot_cases, new_case, tot_death, new_death'
    cases_deaths = client.get("9mfq-cb36", where=query, select=columns, limit=10000)
    df_cases_deaths = pd.DataFrame.from_records(cases_deaths)

    # Normalizes the dates and filters precisely given the cutoff date, in order to keep
    # only data past the beginning of vaccination.
    df_cases_deaths['submission_date'] = df_cases_deaths['submission_date'].apply(
        lambda x: normalize_day(dt.date.fromisoformat(x[0:10])))
    cutoff_date = dt.date.fromisoformat('2020-12-18')
    df_cases_deaths = df_cases_deaths[df_cases_deaths['submission_date'] > cutoff_date]
    df_cases_deaths['submission_date'] = df_cases_deaths['submission_date'].apply(lambda x: str(x))

    # Filters just the 50 states plus D.C.
    abbrev_list = list(abbrev_to_state.keys())
    df_cases_deaths = df_cases_deaths[df_cases_deaths['state'].isin(abbrev_list)]
    df_cases_deaths.reset_index(drop=True, inplace=True)

    # Creates the blank dataframes based on the format frame.
    indexes = df_format.index.copy()
    cols = list(df_format.columns.values)
    df_cases = pd.DataFrame(index=indexes, columns=cols)
    df_cases.fillna(0, inplace=True)
    df_cases['abbrev'] = df_format['abbrev']
    df_deaths = df_cases.copy()

    # Convert string numericals to ints and N/A to 0.
    columns = list(df_cases_deaths.columns.values)[2:]
    for col in columns:
        df_cases_deaths[col] = pd.to_numeric(df_cases_deaths[col])

    # Iterate through the frame and update the formatted frames. Note
    # there are numerous rows for each date and they are summed to get the
    # total for the week ending on the date, which is the column head.
    for row in df_cases_deaths.itertuples():
        index = abbrev_to_state[row[2]]
        if row[4] > df_cases.at[index, row[1]]:
            df_cases.at[index, row[1]] = row[4]
        if row[6] > df_deaths.at[index, row[1]]:
            df_deaths.at[index, row[1]] = row[6]

    return df_cases, df_deaths


def get_fatality_rate(df1, df2):
    '''
    Returns a dataframe of the fatality rates in each state with the dates in question.
    :param df1: DataFrame
    The death information.
    :param df2: DataFrame
    The case information
    :return: DataFrame
    '''
    # Copys format of the frames and calculates the rates.
    indexes = df2.index.copy()
    cols = list(df2.columns.values)
    df_fatality_rate = pd.DataFrame(index=indexes, columns=cols)
    df_fatality_rate['abbrev'] = df1['abbrev']
    df_fatality_rate.iloc[:, 1:] = df1.iloc[:, 1:] / (df2.iloc[:, 1:] +.00001)

    return df_fatality_rate


def get_frame_totals(df, row=True, col=True):
    # Total sum per column:
    if col:
        # df.loc['Total', df.columns[1:]] = df.iloc[:, 1:].sum(axis=0)
        df.loc['Total'] = df.iloc[:, 1:].sum(axis=0)

    # Total sum per row:
    if row:
        df['Total'] = df.iloc[:, 1:].sum(axis=1)

    return df

def get_administered(df_format):
    '''
    Download and clean the CDC data for vaccine doses and 2nd doses administered.
    :param df_format: DataFrame
    The base dataframe (pfizer) used throughout the project.
    :return: tuple
    The two tables (administered and second administered).
    '''

    # Pull the data from github since CDD doesn't have it in their APIs.
    abbrev_to_state = load_pickle('data/state_abbrev.pickle')
    state_to_abbrev = {b: a for a, b in abbrev_to_state.items()}
    url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv'
    df_admin_build = pd.read_csv(url, index_col=0)[['location', 'total_vaccinations', 'people_fully_vaccinated']]

    # Change to our indexes.
    df_admin_build.reset_index(inplace=True)
    df_admin_build.set_index('location', drop=False, inplace=True)


    # Filter by date and change back to string for conversion to column labels.
    df_admin_build['date'] = df_admin_build['date'].apply(
        lambda x: normalize_day(dt.date.fromisoformat(x)))
    cutoff_date = dt.date.fromisoformat('2020-12-18')
    df_admin_build = df_admin_build[df_admin_build['date'] > cutoff_date]
    df_admin_build['date'] = df_admin_build['date'].apply(lambda x: str(x))

    # Change abbreviation column to appropriate form.
    df_admin_build.rename({'location': 'abbrev'}, axis=1, inplace=True)

    # Filters just the 50 states plus D.C.
    abbrev_list = list(abbrev_to_state.keys())
    df_admin_build['abbrev'].replace(state_to_abbrev, inplace=True)
    df_admin_build = df_admin_build[df_admin_build['abbrev'].isin(abbrev_list)]

    # Creates the blank dataframes based on the format frame.
    indexes = df_format.index.copy()
    cols = list(df_format.columns.values)
    df_admin_final = pd.DataFrame(index=indexes, columns=cols)
    df_admin_final.fillna(0, inplace=True)
    df_admin_final['abbrev'] = df_format['abbrev']
    df_second_admin = df_admin_final.copy()

    # Iterate through the frame and update the formatted frames. Note
    # there are numerous rows for each date and they are summed to get the
    # total for the week ending on the date, which is the column head.
    for row in df_admin_build.itertuples():
        index = abbrev_to_state[row[2]]
        if row[3] > df_admin_final.at[index, row[1]]:
            df_admin_final.at[index, row[1]] = row[3]
        if row[4] > df_second_admin.at[index, row[1]]:
            df_second_admin.at[index, row[1]] = row[4]

    return df_admin_final, df_second_admin

def get_admin_totals(df):
    '''
    Returns total column and rows for frames that already possess running total data.
    :param df: DataFrame
    :return: DataFrame
    '''
    # Form total columns and rows (note this is different because the columns are already running totals)
    df['Total'] = df.iloc[:, 1:].max(axis=1)
    df.loc['Total'] = df.iloc[:, 1:].sum(axis=0)

    return df


def drop_zero_cols(df):
    '''
    Drops any columns containing only zeros.
    :param df: DataFrame
    :return: DataFrame
    '''
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def normalize_frames(frame_list):

    len_list = [len(drop_zero_cols(frame).columns) for frame in frame_list]
    min_frame_idx = len_list.index(min(len_list))
    base_frame = drop_zero_cols(frame_list[min_frame_idx])
    base_cols = list(base_frame.columns.values)

    frame_list = [frame[base_cols] for frame in frame_list]

    return frame_list


def add_totals(frame_list):
    '''
    Returns a list of the project's dataframes with 'Total' rows and columns.
    :param frame_list: list
    :return: list
    '''
    # Note this function explicitly assumes a list in the order
    # pfizer, moderna, cases, deaths, fatality rate, administered, 2nd administered.
    # Since some frames need no total (fatality), some are time series, and some are
    # already running totals.
    df1, df2, df3, df4, df5, df6, df7 = frame_list
    df1 = get_frame_totals(df1)
    df2 = get_frame_totals(df2)
    df3 = get_frame_totals(df3)
    df4 = get_frame_totals(df4)
    df5 = get_fatality_averages(df5)
    df6 = get_admin_totals(df6)
    df7 = get_admin_totals(df7)

    return [df1, df2, df3, df4, df5, df6, df7]

def get_fatality_averages(df):

    # Total sum per column:
    df.loc['Total', 1:] = df.iloc[:, 1:].mean(axis=0)

    # Total sum per row:
    df.loc[:, 'Total'] = df.iloc[:, 1:].mean(axis=1)

    return df
