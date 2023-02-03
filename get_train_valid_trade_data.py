import warnings
warnings.filterwarnings("ignore")
import os
from data.yahoodownloader import YahooDownloader
import config.config_tickers as config_tickers
# 01/01/2009 to 09/30/2015
# 10/01/2015 to 12/31/2015
#  01/01/2016 to 05/08/2020
def get_train_valid_trade_data():
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-09-30'
VALID_START_DATE = '2015-10-01'
VALID_END_DATE = '2015-12-31'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2020-05-08'

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

tic_num = len(df.tic.unique())
date_num = len(df.date.unique())
print('{}*{}={}'.format(tic_num, date_num, tic_num*date_num))

from data.preprocessor import FeatureEngineer, data_split
USE_TURBULENCE = False
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=USE_TURBULENCE,
                    user_defined_feature = False)

df = fe.clean_data(df)
df = fe.add_technical_indicator(df)
df = df.fillna(0)
train = data_split(df, TRAIN_START_DATE,TRAIN_END_DATE)
valid = data_split(df, VALID_START_DATE,VALID_END_DATE)
trade = data_split(df, TRADE_START_DATE,TRADE_END_DATE)
trade