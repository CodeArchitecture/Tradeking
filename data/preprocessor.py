def clean_data(df):
    """
    clean the raw data
    deal with missing values
    reasons: stocks could be delisted, not incorporated at the time step
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    # df = data.copy()
    # df = df.sort_values(["date", "tic"], ignore_index=True)
    # df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
    return df

