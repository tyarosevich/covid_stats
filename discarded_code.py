df_vaccine = utils.get_total_frame(df_pfizer, df_moderna)


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

def pfiz_mod_updates(df_old, client):

    df_new_unprocessed = pd.DataFrame.from_records((client.get("saz5-9hgg", limit=1000)))
    df_new_processed = clean_frame(df_new_unprocessed, 'pfizer')
    previous_date = dt.date.fromisoformat(df_old.columns.values[-1])
    new_column = list(df_new_processed.columns.values)[1:]

    update_columns = [x for x in new_column if dt.date.fromisoformat(x) > previous_date]
    for x in update_columns:
        df_old[x] = df_new_processed[x]
    return df_old

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


def get_full_scatter(df1, df2):
    '''
    Returns a px.scatter figure for all vaccine shipments
    and covid deaths in the available data (all states).
    :param df1: DataFrame
    The vaccine shipment frame.
    :param df2: DataFrame
    The covid deaths frame.
    :return: figure
    '''
    # Drop totals, stack rows into a series, and make a dataframe for plotting.
    df_prep = df1.drop(['abbrev', 'Total'], axis=1)
    df_prep.drop('Total', axis=0, inplace=True)
    df_prep = df_prep.cumsum(axis=1, skipna=True)
    x_list = df_prep.stack(dropna=False)
    df_prep2 = df2.drop(['abbrev', 'Total'], axis=1)
    df_prep2.drop('Total', axis=0, inplace=True)
    df_prep2 = df_prep2.cumsum(axis=1, skipna=True)
    y_list = df_prep2.stack(dropna=False)
    s_state = x_list.index.to_list()
    state_list = [tup[0] for tup in s_state]
    df_full_scatter = pd.DataFrame([x_list, y_list]).transpose().rename(columns={0:'x', 1:'y'})
    df_full_scatter['label'] = state_list

    # fig = px.scatter(df_full_scatter,
    #                  x='x',
    #                  y='y')
    fig = go.Figure(data=go.Scattergl(
        x=df_full_scatter['x'],
        y=df_full_scatter['y'],
        mode='markers',
        text=df_full_scatter['label'],
        marker=dict(
            color=np.random.randn(1000),
            colorscale='Blues',
            line_width=1
        )
    ))
    fig.update_layout(
        title='Cumulative Vaccine Shipments and Deaths for all States',
        title_x=0.5,
        xaxis_title='Doses Shipped',
        yaxis_title='Known deaths caused by Covid-19',
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F0F8FF'
    )
    return fig