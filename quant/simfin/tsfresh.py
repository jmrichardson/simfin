
# Process data by ticker
def by_ticker(df, tmp_dir):

    ticker = str(df['Ticker'].iloc[0])
    store = os.path.join(tmp_dir, ticker + ".pickle")
    log.info("Ticker: " + ticker)

    if os.path.isfile(store):
        df = pd.read_pickle(store)
        return df

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    df.reset_index(drop=True, inplace=True)

    for feature in simfinFeatures:
        log.info("  Feature: " + feature)
        if df[feature].count() <= 1:
            log.warning("  Feature count <= 1: " + feature)
            continue

        df_roll, y = make_forecasting_frame(df[feature], kind="price", max_timeshift=16,
                                            rolling_direction=1)
        # X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
        # impute_function=impute, disable_progressbar=True, show_warnings=False)
        X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                             disable_progressbar=True, show_warnings=False)
        X = X.add_prefix('tsfresh_' + feature + '_')
        X = pd.DataFrame().append(pd.Series(), ignore_index=True).append(X, ignore_index=True)
        df = df.join(X)

    log.info("writing " + store)
    df.to_pickle(store)

    return df


def tsfresh_by_ticker(df, tmp_dir)
    log.info("Add tsfresh indicators by ticker ...")
    data = df.groupby('Ticker').apply(by_ticker, tmp_dir)
    return data.reset_index(drop=True, inplace=True)


'''
 log.info("Grouping SimFin data by ticker...")
    data = simfin.groupby('Ticker').apply(by_ticker)
    data.reset_index(drop=True, inplace=True)

    log.info("Saving data ...")
    data.to_pickle("data/quarterly_features_tsfresh.pickle")

    ### Select relevant features
    # Per ticker, pct change, shift by -1 then remove last nan row
    y = simfin.groupby('Ticker')['SPMA'].apply(lambda x: x.pct_change().shift(periods=-1)[:-1])
    y = y.reset_index(drop=True)
    # Per ticker, remove last row because there is nan for y
    X = data.groupby('Ticker').apply(lambda df: df[:-1])
    X = X.loc[:, data.columns.str.startswith('tsfresh_')]
    X = X.reset_index(drop=True)
    X = impute(X)
    new = select_features(X, y)
'''