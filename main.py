import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--window_size",
                    type=int,
                    default=None,
                    help="window size for train trade")
args = parser.parse_args(args=[])

import pandas as pd
import os
if not os.path.exists('results'):
    os.mkdir('results')


df = pd.read_csv('data/df.csv',index_col=0)
# train = pd.read_csv('data/train.csv',index_col=0)
# valid = pd.read_csv('data/valid.csv',index_col=0)
# trade = pd.read_csv('data/trade.csv',index_col=0)

# tech_list = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi_30', 'cci_30', 'dx_30']
tech_list = ['macd_x', 'rsi_30_x', 'cci_30_x', 'dx_30_x', 'macd_y', 'rsi_30_y', 'cci_30_y', 'dx_30_y']

stock_dimension = len(df.tic.unique())
state_space = 1 + 2*stock_dimension + len(tech_list)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

stock_dim = len(df.tic.unique())
from env.PM.env_stocktrading import StockTradingEnv
env_kwargs = {
    "stock_dim": stock_dimension, 
    "hmax": 100, 
    "initial_amount": 1000000, 
    'num_stock_shares': [0] * stock_dimension,
    "buy_cost_pct": 0.001, 
    "sell_cost_pct": 0.001, 
    "reward_scaling": 1e-6,  # 1e-4
    "state_space": state_space, 
    "tech_indicator_list": tech_list,
    "action_space": stock_dimension, 
    'tech_indicator_list': tech_list,
    "print_verbosity":100
}

# It will check your custom environment and output additional warnings if needed
# e_train_gym = StockTradingEnv(df = train, **env_kwargs)
# from stable_baselines3.common.env_checker import check_env
# check_env(e_train_gym)
# e_train_gym = e_train_gym.get_sb_env()


# main
from run_train_trade import run_train_trade

# TOTAL_TIMESTEPS = [100,100,50]
TOTAL_TIMESTEPS = [10000,10000,2000]
run_train_trade(df, env_kwargs, TOTAL_TIMESTEPS, window_size=args.window_size)