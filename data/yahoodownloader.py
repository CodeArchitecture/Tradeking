import pandas as pd
import yfinance as yf


class YahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """
        pd.DataFrame
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, proxy=proxy
            )
            temp_df["tic"] = tic
            data_df = pd.concat([data_df, temp_df])
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        # convert the column names to standardized names
        data_df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjcp",
            "volume",
            "tic",
        ]
        # use adjusted close price instead of close price
        data_df["close"] = data_df["adjcp"]
        # drop the adjusted close price column
        data_df = data_df.drop(labels="adjcp", axis=1)
        # create day of the week column (monday = 0,Tuesday = 1...)
        # data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        # seems useless in this case
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        # sort by date and tic
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        return data_df
