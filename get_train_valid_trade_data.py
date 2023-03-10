from data.yahoodownloader import YahooDownloader
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# 01/01/2009 to 09/30/2015
# 10/01/2015 to 12/31/2015
#  01/01/2016 to 05/08/2020


def get_train_valid_trade_data(
        feature_list,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
        VALID_START_DATE,
        VALID_END_DATE,
        TRADE_START_DATE,
        TRADE_END_DATE,
        tickers):
    L=100
    from datetime import timedelta 
    from datetime import datetime
    # download more date to compute tech indicator like macd
    date = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
    date = date - timedelta(days=L)
    datetime.strftime(date, "%Y-%m-%d")

    df = YahooDownloader(start_date=date,
                         end_date=TRADE_END_DATE,
                         ticker_list=tickers).fetch_data()

    from data.preprocessor import FeatureEngineer, data_split
    from config.config import INDICATORS

    fe = FeatureEngineer(tech_indicator_list=INDICATORS)

    df = fe.clean_data(df)
    df = fe.add_technical_indicator(df)
    df = df.fillna(0)
    # df = fe.add_turbulence(df)
    # df = df.fillna(0)

    df = df[feature_list]
    
    train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    valid = data_split(df, VALID_START_DATE, VALID_END_DATE)
    trade = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    df = data_split(df, TRAIN_START_DATE, TRADE_END_DATE)

    assert len(train.tic.unique())*len(train.date.unique()) == len(train) and \
    len(valid.tic.unique()) * len(valid.date.unique()) == len(valid) and \
    len(trade.tic.unique())*len(trade.date.unique()) == len(trade)
    
#     df = df.loc[L:].reset_index()
#     df = pd.concat([train,trade],ignore_index=True)
    return df, train, valid, trade
