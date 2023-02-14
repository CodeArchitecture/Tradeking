import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gym.Env):
    def __init__(self, 
                df,
                stock_dim,
                action_dim,
                initial_amount,
                transaction_cost_pct,
                tech_indicator_list,
                turbulence_threshold,
                day = 0):

        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.action_dim = action_dim  # if need balance, action_dim = stock_dim+1
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        # self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_dim,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (len(self.tech_indicator_list)*self.stock_dim,),dtype=np.float64)

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        # self.covs = self.data['cov_list'].values[0]
        # self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.state = [self.data[tech].values.tolist() for tech in self.tech_indicator_list]
        self.state = np.array(self.state).flatten()
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_rate_memory = [0]
        self.actions_memory=[[1/self.action_dim]*self.action_dim]
        self.date_memory=[self.data.date.unique()[0]]

        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            pass
            # df = pd.DataFrame(self.portfolio_return_rate_memory)
            # df.columns = ['daily_return']
            # plt.plot(df.daily_return.cumsum(),'r')
            # plt.savefig('results/cumulative_reward.png')
            # plt.close()
            
            # plt.plot(self.portfolio_return_rate_memory,'r')
            # plt.savefig('results/rewards.png')
            # plt.close()

            print("=========one episode end=========")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            # # calculate sharpe ratio
            # df_daily_return = pd.DataFrame(self.portfolio_return_rate_memory)
            # df_daily_return.columns = ['daily_return']
            # if df_daily_return['daily_return'].std() !=0:
            #   sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
            #            df_daily_return['daily_return'].std()
            #   print("Sharpe: ",sharpe)
            print("=================================")
            
            # # if turbulence>self.turbulence_threshold
            # self.state = np.array(self.state).flatten()

            # return self.state, self.reward, self.terminal,{}

        else:
            weights = self.softmax_normalization(actions) 
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            # self.covs = self.data['cov_list'].values[0]
            # self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.state = [self.data[tech].values.tolist() for tech in self.tech_indicator_list]

            self.state = np.array(self.state).flatten()

            transaction_cost_fee = np.sum(np.abs(self.actions_memory[-2]-weights))*self.asset_memory[-1]*self.transaction_cost_pct
            # portfolio_return_rate = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights[1:])
            new_portfolio_value = sum((self.data.close.values / last_day_memory.close.values)*weights*(self.portfolio_value-transaction_cost_fee))
            # print('*'*100, portfolio_return_rate)

            # update portfolio value
            portfolio_return_rate = new_portfolio_value/self.portfolio_value-1
            self.reward = portfolio_return_rate
            # print('*'*100, portfolio_return_rate,self.reward)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_rate_memory.append(portfolio_return_rate)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        # self.covs = self.data['cov_list'].values[0]
        # self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.state = [self.data[tech].values.tolist() for tech in self.tech_indicator_list]
        self.state = np.array(self.state).flatten()
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_rate_memory = [0]
        self.actions_memory=[[1/self.action_dim]*self.action_dim]
        self.date_memory=[self.data.date.unique()[0]] 
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(numerator)
        softmax_output = numerator/denominator
        return softmax_output

    
    # def save_asset_memory(self):
    #     date_list = self.date_memory
    #     portfolio_return_rate = self.portfolio_return_rate_memory
    #     #print(len(date_list))
    #     #print(len(asset_list))
    #     df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return_rate})
    #     return df_account_value

    # def save_action_memory(self):
    #     # date and close price length must match actions length
    #     date_list = self.date_memory
    #     df_date = pd.DataFrame(date_list)
    #     df_date.columns = ['date']
        
    #     action_list = self.actions_memory
    #     df_actions = pd.DataFrame(action_list)
    #     df_actions.columns = self.data.tic.values
    #     df_actions.index = df_date.date
    #     #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
    #     return df_actions

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()   # observation
        return e