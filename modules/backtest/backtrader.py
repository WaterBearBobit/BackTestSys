
from modules.utils import load_first_parquet, load_all_parquets
import pandas as pd
import numpy as np
from modules.config.config import BaseConfig, HFTConfig


class BacktestEngine:
    def __init__(self, data: pd.DataFrame):
        """
        初始化回测引擎
        :param data: pd.DataFrame，数据包含 columns=['date', 'stk_id', 'return', 'buy_sell']
        :param config: BaseConfig，包含回测参数的配置类实例
        """
        self.df = data
        self.config = BaseConfig()  # 配置参数

    def set_date(self, start_date, end_date):
        self.config.start_date = start_date
        self.config.end_date = end_date

    def calculate_strategy_return(self, consider_commission=True, consider_slippage=True):
        # 筛选日期范围内的数据
        self.df = self.df[(self.df['date'] >= self.config.start_date) & (self.df['date'] <= self.config.end_date)]
        
        # 按日期排序
        self.df.sort_values(by=['stk_id', 'date'], inplace=True)
        
        # 计算信号延迟一天生效
        self.df['effective_signal'] = self.df.groupby('stk_id')['buy_sell'].shift(1)
        
        # 计算策略收益率
        self.df['strategy_return'] = np.where(
            self.df['effective_signal'] == 1,
            self.df['return'] - self.config.commission_rate * consider_commission - self.config.slippage * consider_slippage,
            np.where(
                self.df['effective_signal'] == -1,
                -self.df['return'] - self.config.commission_rate * consider_commission - self.config.slippage * consider_slippage,
                0
            )
        )
        
        # 计算累积收益率
        self.df['cumulative_return'] = (1 + self.df['strategy_return']).cumprod() - 1
        
        # 将无效信号的策略收益率置为0
        self.df['strategy_return'] = self.df['strategy_return'].fillna(0)
        
        return self.df

    def backtest(self, consider_commission=True, consider_slippage=True):
        return self.calculate_strategy_return(consider_commission, consider_slippage)

    def backtest_no_commission(self):
        return self.backtest(consider_commission=False)

    def backtest_no_slippage(self):
        return self.backtest(consider_slippage=False)

    def backtest_no_commission_no_slippage(self):
        return self.backtest(consider_commission=False, consider_slippage=False)


class HFTBacktestEngine:
    def __init__(self,  strategy):
        self.config = HFTConfig()
        self.strategy = strategy
        self.portfolio = {'cash': self.config.initial_capital, 'position': 0}  # 投资组合
        self.orders = []  # 交易订单

    def execute_trades(self, signals_df):
        """
        执行交易：根据信号进行买卖操作
        """
        self.portfolio = {'cash': self.config.initial_capital, 'position': 0}  # 投资组合
        self.orders = []  # 订单记录
        pnl_over_time = []  # 记录每个时间点的累计PnL

        for idx, row in signals_df.iterrows():
            signal = row['signal']
            price = row['latestPrice']
            
            if signal == 1 and self.portfolio['cash'] >= price:
                # 买入操作
                self.portfolio['position'] += 1
                self.portfolio['cash'] -= price
                self.orders.append(f"Buy at {row['exchangeTsNanos']} for {price}")
            elif signal == 0 and self.portfolio['position'] > 0:
                # 卖出操作
                self.portfolio['position'] -= 1
                self.portfolio['cash'] += price
                self.orders.append(f"Sell at {row['exchangeTsNanos']} for {price}")

            # 计算当前时间点的累计PnL
            current_value = self.portfolio['cash'] + self.portfolio['position'] * price
            initial_value = self.config.initial_capital
            current_pnl = current_value - initial_value
            pnl_over_time.append(current_pnl)

        return pnl_over_time

    def run_backtest(self):
        """
        运行回测
        """
        # 选择要加载的因子文件（基础因子或超级因子）
        factor_df = load_first_parquet(self.config.solo_factor_output_folder)  # 加载基础因子
        factor_with_signals = self.strategy.generate_signals(factor_df)  # 生成信号
        pnl_over_time = self.execute_trades(factor_with_signals)

        # 返回与原始数据长度相同的PnL序列
        return pnl_over_time
    
    def run_assembled_backtest(self):
        """
        运行回测
        """
        pass



