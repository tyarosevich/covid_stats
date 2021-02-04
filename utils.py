import pickle
import re
import datetime as dt
import pandas as pd

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
    Converts the CDI API output labels to timestamps
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
    timestamp = dt.date.fromisoformat(timestamp)
    if shift:
        timestamp += dt.timedelta(days=5)
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
    # Drops unnecessary columns, and renames date columns to a string timestamp
    # representing 'week ending on this day'. All dates forced up to the next saturday.
    if type == 'covid_deaths':
        # Drop unneeded columns and drop rows from before vaccine distribution began.
        df['mmwrweek'] = pd.to_numeric(df['mmwrweek'])
        df = df[df['mmwrweek'] > 47]
        columns = list(df.columns.values)
        keep_columns = columns[0:4] + columns[17:19]
        df = df[keep_columns]
        df.reset_index(inplace=True, drop=True)
        df.fillna(0, inplace=True)
        columns = list(df.columns.values)
        int_columns = columns[1:3] + columns[4:]
        df[int_columns] = df[int_columns].apply(pd.to_numeric)
        df['week_ending_date'] = df['week_ending_date'].apply(
            dt.date.fromisoformat)
        return df
    columns = list(df.columns.values)
    if type == 'pfizer':
        columns_keep = ['jurisdiction', 'first_doses_12_14'] + [x for x in columns if x[0:8] == 'doses_al' or x[0:8] == 'doses_di']
    elif type == 'moderna':
        columns_keep = ['jurisdiction'] + [x for x in columns if x[0:8] == 'doses_al' or x[0:8] == 'doses_di']

    df = df[columns_keep]
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

#%% More or less ready for Spearman correlation and automating data updates.