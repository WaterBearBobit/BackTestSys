import pandas as pd
import numpy as np
from modules.config.config import BaseConfig, HFTConfig

class Reversal:
    def __init__(self, data):
        self.df = data

    def buy_sell_points(self):
        if 'signal' not in self.df:
            raise ValueError("DataFrame must contain a 'signal' column")
        
        self.df.reset_index(drop=True, inplace=True)
        self.df['factor'] = self.df['signal']

        for stk_id, group in self.df.groupby('stk_id'):
            if group.empty:
                continue  # 如果是空的，则跳过当前循环

            if 'signal' not in group.columns or group['signal'].dropna().empty:
                continue  # 如果'signal'列不存在或全是NaN，则跳过当前循环

            group['signal_rank'] = group['signal'].rank(method='min', ascending=False)
            
            if group['signal_rank'].notna().any():
                threshold = np.percentile(group['signal_rank'].dropna(), 50)
                # 接下来是您的逻辑...
            else:
                print("都是Nan")
                continue  # 或者您也可以跳过这个分组
            
            group['buy_sell'] = (group['signal_rank'] >= threshold).astype(int) * 2 - 1
            
            self.df.loc[group.index, 'buy_sell'] = group['buy_sell']
        
        self.df = self.df[['stk_id', 'date', 'buy_sell', 'return', 'factor']]
        
        return self.df

class Bearish_Engulfing:
    def __init__(self, data):
        self.df = data

    def strategy_bearish_engulfing(self, group, idx, parameter=0):
        """
        判断是否满足看跌吞没形态并返回信号。
        
        参数:
        group (DataFrame): 包含单个股票K线数据的DataFrame，需包含'open'和'close'列。
        idx (int): 当前K线的索引。
        parameter (float): 策略强度的阈值，默认值为 0.001。
        
        返回:
        bool: 如果触发信号返回 True，否则返回 False。
        """
        if idx < 1:  # 无法取到前一根K线时直接返回 False
            return False
        
        # print("group", group)
        # 检查看跌吞没形态条件
        if (group.iloc[idx-1].close > group.iloc[idx-1].open and  # 前一根K线为阳线
            group.iloc[idx].close < group.iloc[idx].open and      # 当前K线为阴线
            group.iloc[idx].open > group.iloc[idx-1].close and   # 当前开盘高于前收盘
            group.iloc[idx].close < group.iloc[idx-1].open):     # 当前收盘低于前开盘
            
            # 计算策略强度
            strength = (np.log(group.iloc[idx].open / group.iloc[idx-1].close) +
                        np.log(group.iloc[idx-1].open / group.iloc[idx].close))
            return strength > parameter
        
        return False
    
    def buy_sell_points(self):
        """
        根据给定的策略为每根K线生成买卖信号。
        """
        self.df = self.df.set_index(['stk_id', 'date'])
        signals = []  # 存储所有股票的信号
        for stk_id, group in self.df.groupby(level='stk_id'):
            group_signals = []  # 存储单个股票的信号
            for idx in range(1, len(group)):
                if self.strategy_bearish_engulfing(group, idx):
                    group_signals.append(-1)  # 发出卖出信号
                else:
                    group_signals.append(1)  # 无信号或持有
            signals.append(pd.Series(group_signals, index=group.index[1:], name=stk_id))
        
        # 将所有股票的信号合并为一个DataFrame
        signals_df = pd.concat(signals, axis=1).fillna(0)
        
        # 将信号DataFrame添加到原始DataFrame中
        self.df = pd.concat([self.df, signals_df], axis=1)
        self.df.columns = list(self.df.columns[:-1]) + ['buy_sell']  # 重命名最后一列为'buy_sell'
        self.df['factor'] = self.df['buy_sell']
        self.df.reset_index(inplace=True)
        self.df = self.df[['stk_id', 'date', 'return', 'buy_sell', 'factor']]

        return self.df
    
    # def buy_sell_points(self):
        """
        根据给定的策略为每根K线生成买卖信号。
        """
        self.df = self.df.set_index(['stk_id', 'date'])
        signals = []  # 存储所有股票的信号
        for stk_id, group in self.df.groupby(level='stk_id'):
            group_signals = []  # 存储单个股票的信号
            for idx in range(1, len(group)):
                if self.strategy_bearish_engulfing(group, idx):
                    group_signals.append(-1)  # 发出买入信号
                else:
                    group_signals.append(1)  # 无信号或持有
            signals.append(pd.Series(group_signals, index=group.index[1:], name=stk_id))
        
        signals_df = pd.concat(signals, axis=1).fillna(0)
        
        # 将信号DataFrame添加到原始DataFrame中
        self.df = pd.concat([self.df, signals_df], axis=1)
        self.df.columns = list(self.df.columns[:-1]) + ['buy_sell']  # 重命名最后一列为'buy_sell'
        self.df['factor'] = self.df['buy_sell']
        self.df.reset_index(inplace=True)
        self.df = self.df[['stk_id', 'date', 'return','buy_sell', 'factor']]

        return self.df

# class Bearish_Engulfing:
#     def __init__(self, data):
#         self.df = data

#     def strategy_bearish_engulfing(df, idx, parameter=1e-3):
#         """
#         判断是否满足看跌吞没形态并返回信号。
        
#         参数:
#         df (DataFrame): 包含股票K线数据，需包含'open'和'close'列。
#         idx (int): 当前K线的索引。
#         parameter (float): 策略强度的阈值，默认值为 0.001。

#         返回:
#         bool: 如果触发信号返回 True，否则返回 False。
#         """
#         if idx < 1:  # 无法取到前一根K线时直接返回 False
#             return False
        
#         # 检查看跌吞没形态条件
#         if (df.iloc[idx-1].close > df.iloc[idx-1].open and  # 前一根K线为阳线
#             df.iloc[idx].close < df.iloc[idx].open and      # 当前K线为阴线
#             df.iloc[idx].open > df.iloc[idx-1].close and   # 当前开盘高于前收盘
#             df.iloc[idx].close < df.iloc[idx-1].open):     # 当前收盘低于前开盘
            
#             # 计算策略强度
#             strength = (np.log(df.iloc[idx].open / df.iloc[idx-1].close) +
#                         np.log(df.iloc[idx-1].open / df.iloc[idx].close))
#             return strength > parameter
        
#         return False

#     def buy_sell_points(self):
#         """
#         根据给定的策略为每根K线生成买卖信号。
        
#         参数:
#         df (DataFrame): 包含股票K线数据。
#         strategy (function): 策略函数，返回 True/False 信号。

#         返回:
#         DataFrame: 原始数据框新增一列'buy_sell'，表示买卖信号。
#         """
#         buy_sell = []
#         for idx in range(len(self.df)):
#             if self.strategy_bearish_engulfing(self.df, idx):
#                 buy_sell.append(-1)  # 卖出信号
#             else:
#                 buy_sell.append(1)   # 买入信号
#         self.df['buy_sell'] = buy_sell
#         return self.df



class Strategy:
    def __init__(self, data, strategy = None):
        self.df = data
        if strategy == 'Bearish_Engulfing':
            self.strategy = Bearish_Engulfing(self.df)
        elif strategy == 'Reversal':
            self.strategy = Reversal(self.df)
        elif strategy == None:
            print('No strategy specified')
        else:
            print(f'Strategy {strategy} have not implemented!') 
    
    def generate_signals(self):
        """
        生成买卖信号（基于因子或者超级因子）
        """
        return self.strategy.buy_sell_points()








class HFTStrategy:
    def __init__(self):
        self.config = HFTConfig()

    def generate_signals(self, factor_df):
        """
        生成买卖信号（基于因子或者超级因子）
        """
        signals = []
        factor_name = factor_df.columns[4]
        
        # 将信号添加到DataFrame
        factor_df['signal'] = factor_df[factor_name] >= 1
        return factor_df[['symbolId', 'exchangeTsNanos', 'latestPrice', 'signal']]  # 保留前三列，丢弃因子列

    def generate_super_factor_signals(self, super_factor_df):
        """
        生成基于超级因子的买卖信号
        """
        pass
