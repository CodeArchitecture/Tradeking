import pandas as pd
from stockstats import StockDataFrame as Sdf

def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


class FeatureEngineer:
    def __init__(
        self,
        tech_indicator_list=None,
    ):
        self.tech_indicator_list = tech_indicator_list

    def clean_data(self, df):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        df = df.reset_index(drop=True)
        df.index = df.date.factorize()[0]
        print('**************data clean completed**************')
        return df

    def add_technical_indicator(self, df):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        stock = df.copy()
        # df = df.sort_values(by=["date", "tic"])
        # df = df.reset_index(drop=True)
        stock = Sdf.retype(stock)
        stock = stock.reset_index()
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator.reset_index(inplace=True)
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["date"] = df[df.tic == unique_ticker[i]]["date"].to_list()
                indicator_df = indicator_df.append(temp_indicator, ignore_index=True)
            df = df.merge(indicator_df[["date", "tic", indicator]], on=["date", "tic"], how="left")

        df = df.sort_values(by=["date", "tic"])
        df.index = df.date.factorize()[0]
        print('************add indicators completed************')
        return df

