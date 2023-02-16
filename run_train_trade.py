from env.PM.env_stocktrading import StockTradingEnv
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from config.config import A2C_PARAMS, PPO_PARAMS, DDPG_PARAMS
from config.config import TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TRADE_START_DATE, TRADE_END_DATE
from data.preprocessor import data_split
import pandas as pd
import numpy as np

def run_train_trade(df, env_kwargs, TOTAL_TIMESTEPS, window_size):
    train = data_split(df,TRAIN_START_DATE,TRAIN_END_DATE)
    train_end_day = train.index[-1]
    trade_end_day = df.index[-1]
    day = 0 # current day
    train_len = train_end_day-day


    for day in range(0, trade_end_day-train_len-window_size, window_size):
        # train phase
        train = df.loc[day:day+train_len]
        train.index = train['date'].factorize()[0]
        e_train_gym = StockTradingEnv(df=train, **env_kwargs).get_sb_env()

        model_a2c, model_ppo, model_ddpg = run_train(e_train_gym,TOTAL_TIMESTEPS)

        models={'a2c':model_a2c,'ppo':model_ppo,'ddpg':model_ddpg}
        # trade phase
        trade = df.loc[day+train_len+1:day+train_len+window_size]
        trade.index = trade['date'].factorize()[0]
        # print(trade)

        for model_name in ['a2c','ppo','ddpg']:
            e_trade_gym = StockTradingEnv(df = trade, **env_kwargs).get_sb_env()
            rewards = run_trade(e_trade_gym, models, model_name)
            pd.Series(rewards).to_csv('results/asset_value_{}_{}.csv'.format(model_name,day))

    return None

        
def run_train(e_train_gym, TOTAL_TIMESTEPS):
    print('A2C training start')
    model_a2c = A2C("MlpPolicy", e_train_gym, verbose=0, **A2C_PARAMS)
    model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS[0])
    print('A2C training end')
    print('PPO training start')
    model_ppo = PPO("MlpPolicy", e_train_gym, verbose=0, **PPO_PARAMS)
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS[1])
    print('PPO training end')
    print('DDPG training start')
    model_ddpg = DDPG("MlpPolicy", e_train_gym, verbose=0, **DDPG_PARAMS)
    model_ddpg.learn(total_timesteps=TOTAL_TIMESTEPS[2])
    print('DDPG training end')
    return model_a2c, model_ppo, model_ddpg


def run_trade(e_trade_gym, models, model_name):
    model = models[model_name]
    rewards=[]  # 
    dones = False
    obs = e_trade_gym.reset()
    while not dones:
        action, _states = model.predict(obs)
        # print(_states)
        # print(action)
        obs, reward, dones, info = e_trade_gym.step(action)
        rewards.append(float(reward))

    return rewards