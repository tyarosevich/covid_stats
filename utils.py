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

    # Drops unnecessary columns, and renames date columns to a string timestamp
    # representing 'week ending on this day'. All dates forced up to the next saturday.
    if type == 'covid_deaths':
        # Drop unneeded columns and drop rows from before vaccine distribution began.
        df['week_ending_date'] = df['week_ending_date'].apply(
            dt.date.fromisoformat)
        df['mmwrweek'] = pd.to_numeric(df['mmwrweek'])
        df = df[df['week_ending_date'] > dt.date.fromisoformat('2020-12-18')]
        columns = list(df.columns.values)
        try:
            keep_columns = columns[0:4] + columns[17:19]
        except IndexError:
            print("Source columns have changed")
        try:
            df = df[keep_columns]
        except KeyError:
            print("Source columns have changed.")
        df.reset_index(inplace=True, drop=True)
        df.fillna(0, inplace=True)
        columns = list(df.columns.values)
        int_columns = columns[1:3] + columns[4:]
        df[int_columns] = df[int_columns].apply(pd.to_numeric)
        df['week_ending_date'] = df['week_ending_date'].apply(normalize_day)
        return df

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


def correct_bad_aggreg(df_old, df_format):
    state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut",
                   "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois",
                   "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan",
                   "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska",
                   "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon",
                   "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
                   "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

    # Format the new df using df_format as a template.
    col_labels = list(df_format.columns.values)
    df_new = pd.DataFrame(columns=col_labels)
    df_new['abbrev'] = df_format['abbrev']

    # Create sets to make sure old format has the data.
    state_set = set(df_format['abbrev'])
    date_set = set(list(df_format.columns.values)[1:])

    # iterate through the old frame to update the new one.
    for row in df_old.itertuples():
        # Get the index of the state in the new, good format frame.
        if row[1] in state_set and str(row[4]) in date_set:
            # x = df_new[df_new['state'] == row[1]].index.values.astype(int)[0]
            df_new[str(row[4])][row[1]] = row[5]
    df_new = df_new.loc[state_names, :]
    return df_new

def update_frames():

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
    if day_of_week == 0 and flag ==0:
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

        # Reformat to needed columns, and column labels that are datetime ready
        # strings. Column labels are all moved to 'week ending on' the date, which is
        # a saturday, to conform to other CDC data.
        df_pfizer_formatted = clean_frame(df_pfizer, 'pfizer')
        df_moderna_formatted = clean_frame(df_moderna, 'moderna')

        # Totally clean, re-format and re-aggregate this disaster of a dataset. Pfizer frame
        # is passed as template.
        # df_covid_deaths_formatted = correct_bad_aggreg(clean_frame(df_covid_deaths, 'covid_deaths'),
                                                             # df_pfizer_formatted)

        df_cases, df_deaths = get_case_deaths(df_pfizer_formatted)
        df_fatality_rate = get_fatality_rate(df_deaths, df_cases)
        frame_list = [df_pfizer_formatted, df_moderna_formatted, df_cases, df_deaths, df_fatality_rate]
        # Save formatted data
        save_pickle('data/formatted_data.pickle', frame_list)

        return frame_list
    # If it's not monday, flag for update.
    else:
        flag = 0
        save_pickle('data/flag.pickle', flag)
        frame_list = load_pickle('data/formatted_data.pickle')
        return frame_list


def pfiz_mod_updates(df_old, client):

    df_new_unprocessed = pd.DataFrame.from_records((client.get("saz5-9hgg", limit=1000)))
    df_new_processed = clean_frame(df_new_unprocessed, 'pfizer')
    previous_date = dt.date.fromisoformat(df_old.columns.values[-1])
    new_column = list(df_new_processed.columns.values)[1:]

    update_columns = [x for x in new_column if dt.date.fromisoformat(x) > previous_date]
    for x in update_columns:
        df_old[x] = df_new_processed[x]
    return df_old

def covid_deaths_updates(df_old, df_format, client):

    today = dt.date.today()
    day_of_week = today.weekday()
    holder_array = df_old.iloc[0][1:].to_numpy()
    # Have to iterate through the array because something is bugging out. This gets
    # the row index for the last column whose first entry has something.
    last_col_idx = 0
    for j, x in enumerate(holder_array):
        if np.isnan(x):
            last_col_idx = j
            break
    last_date = dt.date.fromisoformat(list(df_old.columns.values)[last_col_idx])
    norm_date_str = str(last_date + dt.timedelta(days=7))
    # Get previous week too in case some states are behind.
    query = 'week_ending_date = "{}" or week_ending_date = "{}"'.format(str(norm_date_str), str(last_date))
    # %% update for covid_deaths
    # Get jsons from CDC
    # results_pfizer = client.get("saz5-9hgg", limit=1000)
    # results_moderna = client.get("b7pe-5nws", limit=1000)
    covid_deaths = client.get("muzy-jte6", where=query, limit=4000)
    df_new = pd.DataFrame.from_records(covid_deaths)

    df_new_deaths_formatted = correct_bad_aggreg(clean_frame(df_new, 'covid_deaths'), df_format)

    # Add any new columns to the permanent dataframe.
    for col in list(df_new_deaths_formatted.columns.values)[1:]:
        if col not in list(df_old.columns.values):
            df_old[col] = np.nan

    # Update any nan values in the permanent dataframe.
    df_old.update(df_new_deaths_formatted, overwrite=False, errors='ignore')

    return df_old

def get_current_data():
    # Check for updates every monday
    today = dt.date.today()
    day_of_week = today.weekday()
    formatted_list = load_pickle('data/formatted_data.pickle')
    if day_of_week == 0:
        client = Socrata("data.cdc.gov", None)
        updated_list = update_frames(formatted_list, client)
        # Save updated data.
        save_pickle('data/processed_data.pickle', updated_list)
        return updated_list
    else:
        return formatted_list

def get_total_frame(df1, df2):

    # Combine the two dataframes and replace states with abbrevs.
    df_vaccine = df1
    df_vaccine.iloc[:, 2:] += df2.iloc[:, 1:]
    abbrev_to_state = load_pickle('data/state_abbrev.pickle')
    state_to_abbrev = {b: a for a, b in abbrev_to_state.items()}
    # df_vaccine['state'].replace(state_to_abbrev, inplace=True)


    # Total sum per column:
    df_vaccine.loc['Total', df_vaccine.columns[1:]] = df_vaccine.iloc[:, 1:].sum(axis=0)

    # Total sum per row:
    df_vaccine.loc[:, 'Total'] = df_vaccine.iloc[:, 1:].sum(axis=1)

    # Fix stupid float bug.
    cols = list(df_vaccine.columns.values)
    cols = cols[1:]
    type_dict = dict(zip(cols, ['int64'] * len(cols)))
    df_vaccine = df_vaccine.astype(type_dict)

    return df_vaccine


def total_rows_columns(df):
    # Total sum per column:
    df.loc['Total', 1:] = df.iloc[:, 1:].sum(axis=0)

    # Total sum per row:
    df.loc[:, 'Total'] = df.iloc[:, 1:].sum(axis=1)

    return df

def get_df_pop_and_deaths(df1, df2, df3):
    # df1 = vaccine, df2 = population, df3 = deaths
    df_vacc_withpop = df1.merge(df2, how='inner')

    df_vacc_withpop.drop('abbrev', axis=1, inplace=True)
    df_vacc_withpop['per1000'] = df_vacc_withpop['Total'] / df_vacc_withpop['2019'] * 1000
    df_deaths_wtotals = total_rows_columns(df3)
    df_totals_asframe = df_deaths_wtotals['Total'].to_frame()

    df_totals_asframe['Total_deaths'] = df_totals_asframe['Total'].astype('int32')

    df_vacc_withpop_and_deaths = df_vacc_withpop.merge(df_totals_asframe, how='inner', left_index=True,
                                                       right_index=True)

    # Create a column for doses per death
    df_vacc_withpop_and_deaths['relative'] = df_vacc_withpop_and_deaths['Total_x'] / (df_vacc_withpop_and_deaths['Total_deaths'] + .01)
    return df_vacc_withpop_and_deaths

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
        df_cases.loc[index, row[1]] += row[4]
        df_deaths.loc[index, row[1]] += row[6]

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
    df_fatality_rate.iloc[:, 1:] = df1.iloc[:, 1:] / df2.iloc[:, 1:]

    return df_fatality_rate


def get_frame_totals(df, row=True, col=True):
    # Total sum per column:
    if col:
        df.loc['Total', df.columns[1:]] = df.iloc[:, 1:].sum(axis=0)

    # Total sum per row:
    if row:
        df.loc[:, 'Total'] = df.iloc[:, 1:].sum(axis=1)

    return df