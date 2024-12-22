import pandas as pd
import numpy as np
import logging
from modules.config.config import HFTConfig, DailyConfig, BaseConfig

# Define individual factor classes

class StkDailyFactors:
    def __init__(self, data: pd.DataFrame):
        """
        初始化股票日频因子计算类
        :param data: 股票日频数据 DataFrame
        """
        self.data = data

    def compute_factors_for_group(self, group: pd.DataFrame):
        """
        对单个股票分组计算因子
        """
        # group["momentum_5d"] = group["close"].pct_change(periods=5)
        # group["volatility_10d"] = group["close"].rolling(window=10).std()
        # group["volume_change_5d"] = group["volume"].pct_change(periods=5)
        group["reversal_5d"] = -group["close"].pct_change(periods=5)
        group["reversal_10d"] = -group["close"].pct_change(periods=10)
        group["reversal_15d"] = -group["close"].pct_change(periods=15)
        group["reversal_30d"] = -group["close"].pct_change(periods=30)
        group["return"] = group["close"].pct_change()
        return group

    def get_factors(self):
        """
        计算并返回包含所有计算后的因子的 DataFrame
        """
        # 按股票ID分组，并在每个分组上计算因子
        self.data = self.data.groupby('stk_id').apply(self.compute_factors_for_group)
        return self.data

# # 股票日频因子计算类
# class StkDailyFactors:
#     def __init__(self, data: pd.DataFrame):
#         """
#         初始化股票日频因子计算类
#         :param data: 股票日频数据 DataFrame
#         """
#         self.data = data

#     def compute_momentum_5d(self):
#         """
#         计算5日动量因子
#         """
#         self.data["momentum_5d"] = self.data["close"].pct_change(periods=5)

#     def compute_volatility_10d(self):
#         """
#         计算10日波动率因子
#         """
#         self.data["volatility_10d"] = self.data["close"].rolling(window=10).std()

#     def compute_volume_change_5d(self):
#         """
#         计算5日成交量变化率因子
#         """
#         self.data["volume_change_15d"] = self.data["volume"].pct_change(periods=5)

#     def compute_reversal_5d(self):
#         """
#         计算5日反转因子
#         """
#         # 反转因子可以是动量因子的相反数，即过去5天价格的负变化率
#         self.data["reversal_5d"] = -self.data["close"].pct_change(periods=5)

#     def compute_reversal_10d(self):
#         """
#         计算10日反转因子
#         """
#         # 反转因子可以是动量因子的相反数，即过去10天价格的负变化率
#         self.data["reversal_10d"] = -self.data["close"].pct_change(periods=10)

#     def compute_reversal_15d(self):
#         """
#         计算15日反转因子
#         """
#         # 反转因子可以是动量因子的相反数，即过去15天价格的负变化率
#         self.data["reversal_15d"] = -self.data["close"].pct_change(periods=15)

#     def compute_reversal_30d(self):
#         """
#         计算30日反转因子
#         """
#         # 反转因子可以是动量因子的相反数，即过去30天价格的负变化率
#         self.data["reversal_30d"] = -self.data["close"].pct_change(periods=30)

#     def get_factors(self):
#         """
#         计算并返回包含所有计算后的因子的 DataFrame
#         """
#         self.compute_momentum_5d()
#         self.compute_volatility_10d()
#         self.compute_volume_change_5d()
#         self.compute_reversal_5d()
#         self.compute_reversal_10d()
#         self.compute_reversal_15d()
#         self.compute_reversal_30d()
#         return self.data


# 财务注释因子计算类
class StkFinAnnotationFactors:
    def __init__(self, data: pd.DataFrame):
        """
        初始化财务注释因子计算类
        :param data: 股票财务注释数据 DataFrame
        """
        self.data = data

    def compute_nonrecurring_total(self):
        """
        计算非经常性损益总和
        """
        self.data["nonrecurring_total"] = self.data.filter(like="NONRECURRING_").sum(axis=1)

    def get_factors(self):
        """
        计算并返回包含计算后的财务注释因子的 DataFrame
        """
        self.compute_nonrecurring_total()
        return self.data


# 财务平衡表因子计算类
class StkFinBalanceFactors:
    def __init__(self, data: pd.DataFrame):
        """
        初始化财务平衡表因子计算类
        :param data: 股票财务平衡表数据 DataFrame
        """
        self.data = data

    def compute_debt_to_asset(self):
        """
        计算资产负债比
        """
        self.data["debt_to_asset"] = self.data["BALANCESTATEMENT_10"] / (self.data["BALANCESTATEMENT_11"] + 2 ** (-10))

    def get_factors(self):
        """
        计算并返回包含计算后的财务平衡表因子的 DataFrame
        """
        self.compute_debt_to_asset()
        return self.data


# 财务现金流因子计算类
class StkFinCashflowFactors:
    def __init__(self, data: pd.DataFrame):
        """
        初始化财务现金流因子计算类
        :param data: 股票财务现金流数据 DataFrame
        """
        self.data = data

    def compute_operating_cashflow_ratio(self):
        """
        计算经营活动现金流占比
        """
        self.data["operating_cashflow_ratio"] = self.data["CASHFLOWSTATEMENT_10"] / self.data["adj"]

    def get_factors(self):
        """
        计算并返回包含计算后的财务现金流因子的 DataFrame
        """
        self.compute_operating_cashflow_ratio()
        return self.data


# 财务收入表因子计算类
class StkFinIncomeFactors:
    def __init__(self, data: pd.DataFrame):
        """
        初始化财务收入表因子计算类
        :param data: 股票财务收入表数据 DataFrame
        """
        self.data = data

    def compute_net_profit_margin(self):
        """
        计算净利润率
        """
        self.data["net_profit_margin"] = self.data["INCOMESTATEMENT_10"] / self.data["INCOMESTATEMENT_9"]

    def get_factors(self):
        """
        计算并返回包含计算后的财务收入表因子的 DataFrame
        """
        self.compute_net_profit_margin()
        return self.data


# DailyComputeEngine class that selects appropriate factors class based on data_type
class DailyComputeEngine:
    def __init__(self, data: pd.DataFrame, data_type: str = "stk_daily"):
        """
        初始化计算引擎
        
        :param data: 数据，包含一个 DataFrame
        :param data_type: 数据类型（例如 "stk_daily", "stk_fin_annotation" 等）
        """
        self.config = DailyConfig()
        
        # 根据数据类型选择相应的因子类
        if data_type == "stk_daily":
            self.factors = StkDailyFactors(data)
        elif data_type == "stk_fin_annotation":
            self.factors = StkFinAnnotationFactors(data)
        elif data_type == "stk_fin_balance":
            self.factors = StkFinBalanceFactors(data)
        elif data_type == "stk_fin_cashflow":
            self.factors = StkFinCashflowFactors(data)
        elif data_type == "stk_fin_income":
            self.factors = StkFinIncomeFactors(data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def compute_factors(self):
        """
        计算因子，返回计算结果
        """
        factor_dict = self.factors.get_factors()
        return factor_dict



class HFTComputeEngine:
    def __init__(self, depth=None, order=None, trade=None, rolling = 0):
        """
        Initializes the ComputeEngine with configuration and data.
        
        Parameters:
            config (dict): Configuration dictionary for the engine.
            depth (pd.DataFrame): Depth data.
            order (pd.DataFrame): Order data.
            trade (pd.DataFrame): Trade data.
        """
        self.config = HFTConfig()
        self.depth = depth
        self.order = order
        self.trade = trade
        self.factors = pd.DataFrame()  # Store calculated factors
        self.rolling_sec = rolling

        self.pre_process_depth()        
    
    def pre_process_depth(self):
                # 处理 10 档买卖量价
        self.bid_prices = self.depth["bids"].apply(lambda x: np.array([i["price"] if i["price"] > 0 else np.nan for i in x]))
        self.bid_volumes = self.depth["bids"].apply(lambda x: np.array([i["volume"] for i in x]))
        self.bid_nums = self.depth["bids"].apply(lambda x: np.array([i["no"] for i in x]))

        self.ask_prices = self.depth["asks"].apply(lambda x: np.array([i["price"] if i["price"] > 0 else np.nan for i in x]))
        self.ask_volumes = self.depth["asks"].apply(lambda x: np.array([i["volume"] for i in x]))
        self.ask_nums = self.depth["asks"].apply(lambda x: np.array([i["no"] for i in x]))

        self.depth['lookback_start'] = self.depth.index - pd.Timedelta(seconds=self.rolling_sec)
        self.depth["bid_price_1"] = self.depth['bids'].apply(lambda x: x[0]["price"])
        self.depth["ask_price_1"] = self.depth['asks'].apply(lambda x: x[0]["price"])
        self.depth["bid_vlm_1"] = self.depth['bids'].apply(lambda x: x[0]["volume"])
        self.depth["ask_vlm_1"] = self.depth['asks'].apply(lambda x: x[0]["volume"])

    def calculate_returns(self, intervals):
        """
        Calculate returns for the given intervals.
        
        Parameters:
            intervals (list): List of intervals in seconds.
        """
        for interval in intervals:
            col_label = f"return_{interval}s"
            self.depth[col_label] = self.depth.index + pd.Timedelta(seconds=interval)
            self.depth[col_label] = self.depth.apply(lambda row: self._calculate_single_return(row, interval), axis=1)

    def _calculate_single_return(self, row, interval):
        """
        Internal helper to calculate single return for a given row and interval.
        """
        window = self.depth.loc[row.name:row[f"return_{interval}s"]]
        if len(window) > 2:
            ask_1_first = window["ask_price_1"].values[0]
            bid_1_first = window["bid_price_1"].values[0]
            ask_1_last = window["ask_price_1"].values[-1]
            bid_1_last = window["bid_price_1"].values[-1]
            return ((ask_1_last + bid_1_last) - (ask_1_first + bid_1_first)) / (ask_1_first + bid_1_first)
        return np.nan

    def calculate_raw_features(self):
        def aggregate_row(row):
            window = self.depth.loc[row['lookback_start']:row.name]
            # Bids and Asks data        
            bid_price_1 = window['bid_price_1']
            ask_price_1 = window['ask_price_1']

            bid_vlm_1 = window['bid_vlm_1']
            ask_vlm_1 = window['ask_vlm_1']
            return pd.Series({
                # 'open': window['latestPrice'].iloc[0],
                # 'high': window['latestPrice'].max(),
                # 'low': window['latestPrice'].min(),
                # 'close': window['latestPrice'].iloc[-1],
                # 'volume': window['volume'].sum(),

                'first_bidPx1': bid_price_1.iloc[0],
                'first_askPx1': ask_price_1.iloc[0],
                'max_bidPx1': bid_price_1.max(),
                'max_askPx1': ask_price_1.max(),
                'min_bidPx1': bid_price_1.min(),
                'min_askPx1': ask_price_1.min(),
                'last_bidPx1': bid_price_1.iloc[-1],
                'last_askPx1': ask_price_1.iloc[-1],
                'last_bidVlm1': bid_vlm_1.iloc[-1],
                'last_askVlm1': ask_vlm_1.iloc[-1],
                # 'y_15s': y_15s,
                # 'y_300s': y_300s,
                # 'y_600s': y_600s,
                # 'y_1200s': y_1200s,

                # # Summed volumes and amounts for bids and asks
                # 'last_sumbidVlm': bid_vlm_1.sum(),
                # 'last_sumaskVlm': ask_vlm_1.sum(),
                # 'last_sumbidAmt': (bid_price_1 * bid_vlm_1).sum(),
                # 'last_sumaskAmt': (ask_price_1 * ask_vlm_1).sum(),
                
                # # Max bid and ask volumes
                # 'max_bidVlm1': bid_vlm_1.max(),
                # 'max_askVlm1': ask_vlm_1.max(),
                
                # # Max of bid and ask volumes (overall max in the window)
                # 'max_maxbidVlm': bids.apply(lambda x: max([entry["volume"] for entry in x])).max(),
                # 'max_maxaskVlm': asks.apply(lambda x: max([entry["volume"] for entry in x])).max(),
                
                # # Mean bid and ask prices
                # 'mean_bidPx1': bid_price_1.mean(),
                # 'mean_askPx1': ask_price_1.mean(),
                
                # # Sum of bid and ask amounts
                # 'sum_bidAmt1': (bid_price_1 * bid_vlm_1).sum(),
                # 'sum_askAmt1': (ask_price_1 * ask_vlm_1).sum(),
                
                # # Sum of bid and ask volumes
                # 'sum_bidVlm1': bid_vlm_1.sum(),
                # 'sum_askVlm1': ask_vlm_1.sum(),
                
                # # VWAP (Volume Weighted Average Price) for bids and asks
                # 'vwa_bidPx1': (bid_price_1 * bid_vlm_1).sum() / bid_vlm_1.sum() if bid_vlm_1.sum() != 0 else 0,
                # 'vwa_askPx1': (ask_price_1 * ask_vlm_1).sum() / ask_vlm_1.sum() if ask_vlm_1.sum() != 0 else 0,
                
                # # Standard deviation of volumes for bids and asks
                # 'std_bidVlm1': bid_vlm_1.std(),
                # 'std_askVlm1': ask_vlm_1.std(),
                # 'std_maxbidVlm': bids.apply(lambda x: max([entry["volume"] for entry in x])).std(),
                # 'std_maxaskVlm': asks.apply(lambda x: max([entry["volume"] for entry in x])).std(),
                # 'std_sumbidVlm': bids.apply(lambda x: sum([entry["volume"] for entry in x])).std(),
                # 'std_sumaskVlm': asks.apply(lambda x: sum([entry["volume"] for entry in x])).std(),

                # # Spread calculations (bid - ask)
                # 'mean_spreadp': (ask_price_1 - bid_price_1).mean(),
                # 'max_spreadp': (ask_price_1 - bid_price_1).max(),
            })

            # Apply the aggregation function to each row
        aggregated_df = self.depth.apply(aggregate_row, axis=1)
        
        return aggregated_df

    def S1001(self):
        """
        bid-ask spread
        """
        spread_sr = self.ask_prices.apply(lambda x: x[0]) - self.bid_prices.apply(lambda x: x[0])
        return spread_sr

    def S1002(self):
        """
        mid price
        """
        mid_sr = (self.ask_prices.apply(lambda x: x[0]) + self.bid_prices.apply(lambda x: x[0])) / 2
        return mid_sr

    def S1003(self):
        """
        total bid count
        """
        total_bid_count = self.bid_nums.apply(sum)
        return total_bid_count

    def S1004(self):
        """
        total ask count
        """
        total_ask_count = self.ask_nums.apply(sum)
        return total_ask_count

    def S1005(self):
        """
        total bid volume
        """
        total_bid_count = self.bid_volumes.apply(sum)
        return total_bid_count

    def S1006(self):
        """
        total ask volume
        """
        total_ask_count = self.ask_volumes.apply(sum)
        return total_ask_count

    def S1007(self):
        """
        total bid amount
        """
        bid_amount = (self.bid_prices * self.bid_volumes).apply(lambda x: np.nansum(x))
        return bid_amount

    def S1008(self):
        """
        total ask amount
        """
        ask_amount = (self.ask_prices * self.ask_volumes).apply(lambda x: np.nansum(x))
        return ask_amount

    def S1009(self):
        """
        Order Flow Imbalance (OFI)
        """
        best_bid_prices = self.bid_prices.apply(lambda x: x[0])
        best_ask_prices = self.ask_prices.apply(lambda x: x[0])
        
        best_bid_volumes = self.bid_volumes.apply(lambda x: x[0])
        best_ask_volumes = self.ask_volumes.apply(lambda x: x[0])
        
        bid_price_change = best_bid_prices.diff()
        ask_price_change = best_ask_prices.diff()

        bid_volume_change = best_bid_volumes.diff()
        ask_volume_change = best_ask_volumes.diff()

        # Calculate OFI
        ofi = (
            (bid_price_change < 0) * (-bid_volume_change) +  # Bid price drops: subtract bid volume
            (bid_price_change == 0) * bid_volume_change +   # Bid price unchanged: add bid volume
            (ask_price_change > 0) * (-ask_volume_change) + # Ask price rises: subtract ask volume
            (ask_price_change == 0) * ask_volume_change     # Ask price unchanged: add ask volume
        )

        return ofi

    def S1010(self):
        """
        Correlation between bid_prices and bid_volumes, ignoring NaN values.
        """
        def calculate_correlation(prices, volumes):
            mask = ~np.isnan(prices) & ~np.isnan(volumes)
            filtered_prices = prices[mask]
            filtered_volumes = volumes[mask]

            if len(filtered_prices) > 1:
                mean_x = np.mean(filtered_prices)
                mean_y = np.mean(filtered_volumes)
                
                numerator = np.sum((filtered_prices - mean_x) * (filtered_volumes - mean_y))
                denominator = np.sqrt(np.sum((filtered_prices - mean_x) ** 2) * np.sum((filtered_volumes - mean_y) ** 2))
                
                if denominator != 0:
                    return numerator / denominator
                else:
                    return np.nan
            else:
                return np.nan

        temp_df = pd.DataFrame({
            'bid_prices': self.bid_prices,
            'bid_volumes': self.bid_volumes
        })

        correlation_sr = temp_df.apply(
            lambda row: calculate_correlation(row['bid_prices'], row['bid_volumes']), axis=1
        )

        return correlation_sr

    def S1011(self):
        """
        Correlation between ask_prices and ask_volumes, ignoring NaN values.
        """
        def calculate_correlation(prices, volumes):
            mask = ~np.isnan(prices) & ~np.isnan(volumes)
            filtered_prices = prices[mask]
            filtered_volumes = volumes[mask]

            if len(filtered_prices) > 1:
                mean_x = np.mean(filtered_prices)
                mean_y = np.mean(filtered_volumes)
                
                numerator = np.sum((filtered_prices - mean_x) * (filtered_volumes - mean_y))
                denominator = np.sqrt(np.sum((filtered_prices - mean_x) ** 2) * np.sum((filtered_volumes - mean_y) ** 2))
                
                if denominator != 0:
                    return numerator / denominator
                else:
                    return np.nan
            else:
                return np.nan

        temp_df = pd.DataFrame({
            'ask_prices': self.ask_prices,
            'ask_volumes': self.ask_volumes
        })

        correlation_sr = temp_df.apply(
            lambda row: calculate_correlation(row['ask_prices'], row['ask_volumes']), axis=1
        )

        return correlation_sr
    
        
    def calculate_IC(self):
        """
        Calculate Information Coefficient (IC) for each factor starting with 'S' in the class.
        """
        # Collect all methods starting with 'S' (excluding 'S' methods that don't return series)
        factors = []
        factor_names = []
        
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('S'):
                # Ensure that the method is callable and returns a pandas Series
                try:
                    factor_series = method()
                    if isinstance(factor_series, pd.Series):
                        factors.append(factor_series)
                        factor_names.append(name)
                except Exception as e:
                    logging.warning(f"Failed to compute {name}: {e}")

        # Calculate ICs
        def single_y_ic(y_col):
            y = self.agg_df[y_col]  # Target variable
            y = y.clip(lower=y.quantile(0.1/100), upper=y.quantile(99.9/100))

            ic_df = pd.DataFrame(pd.concat(factors, axis=1).corrwith(y)).T
            ic_df.index = [self.stock_code.rstrip(".xz")] * len(ic_df)
            ic_df.index.name = "stock_code"
            ic_df.columns = factor_names
            return ic_df

        y_cols = ["y_15s", "y_300s", "y_600s", "y_1200s"]
        agg_ic_df = pd.concat([single_y_ic(y_col) for y_col in y_cols])
        agg_ic_df["y"] = y_cols
        agg_ic_df.set_index("y", append=True, inplace=True)

        return agg_ic_df

    def save_factors(self, factor_df):
        """
        将计算的因子数据保存为parquet格式
        """
        factor_df.to_parquet(self.config.generated_factors_path)
        print(f"Factors saved to {self.config.generated_factors_path}")
