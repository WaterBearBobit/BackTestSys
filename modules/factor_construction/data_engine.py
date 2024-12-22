from modules.utils import capnp_to_df
from modules.config.config import HFTConfig, DailyConfig, BaseConfig

import pandas as pd
import numpy as np
import os
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

class DataEngineFactory:
    @staticmethod
    def get_data_engine(data_frequency, raw_data_path):
        if data_frequency == 'high':
            return HFTDataEngine(raw_data_path)
        elif data_frequency == 'low':
            return DailyDataEngine(raw_data_path)
        else:
            raise ValueError("Unsupported frequency type")

from abc import ABC, abstractmethod

class DataEngine(ABC):
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data = None

    def load_data(self):
        # 加载原始数据，通用逻辑
        pass

    def clean_data(self):
        # 通用清洗逻辑，比如去掉无效行、填补缺失值
        pass

    @abstractmethod
    def process_frequency_specific_data(self):
        # 抽象方法，子类必须实现
        pass

    def get_processed_data(self):
        if self.processed_data is None:
            self.clean_data()
            self.process_frequency_specific_data()
        return self.processed_data


class DailyDataEngine:
    def __init__(self, config = DailyConfig()):
        self.config = config
        self.data = {}

    def load_data(self):
        """
        加载所有日频数据文件。
        """
        for key, file_path in self.config.daily_files.items():
            try:
                # print(f"Loading {key} from {file_path}...")
                self.data[key] = pd.read_feather(file_path)
            except Exception as e:
                print(f"Failed to load {key}: {e}")

    def preprocess_data(self):
        """
        对加载的数据进行预处理。
        """
        if "stk_daily" in self.data:
            daily_data = self.data["stk_daily"]
            # 例子：日期格式转换
            daily_data["date"] = pd.to_datetime(daily_data["date"])
            # 例子：缺失值处理
            daily_data.fillna(0, inplace=True)
            self.data["stk_daily"] = daily_data

    def get_data(self, key):
        """
        获取特定的数据。
        """
        return self.data.get(key, None)



class HFTDataEngine(DataEngine):
    def __init__(self, config = HFTConfig()):
        self.config = config        
    
    def split_data(self):
        """
        处理parqute数据
        """
        path = self.config.factor_output_folder
        assembled_path = os.path.join(path, 'assembled')  # 文件夹路径
        splited_path = os.path.join(path, 'solo')  # 保存拆分后的数据的文件夹路径
        
        # 确保目标文件夹存在
        if not os.path.exists(splited_path):
            os.makedirs(splited_path)

        # 读取 assembled_path 文件夹中的 .parquet 文件
        # 假设文件夹中只有一个文件
        files = [f for f in os.listdir(assembled_path) if f.endswith('.parquet')]
        if not files:
            raise FileNotFoundError(f"No parquet files found in {assembled_path}")

        # 选择第一个文件进行处理
        assembled_file = os.path.join(assembled_path, files[0])

        # 读取该文件为 DataFrame
        df = pd.read_parquet(assembled_file)

        # 检查是否包含 'symbolId' 列
        if 'symbolId' not in df.columns:
            raise ValueError("DataFrame does not contain 'symbolId' column")

        # 按照 'symbolId' 列进行分割并保存为单独的 .parquet 文件
        unique_symbol_ids = df['symbolId'].unique()
        for symbol_id in unique_symbol_ids:
            # 根据 'symbolId' 筛选出相应的数据
            symbol_df = df[df['symbolId'] == symbol_id]

            # 创建保存文件的路径
            symbol_file = os.path.join(splited_path, f"{symbol_id}_factor.parquet")
            
            # 保存为 parquet 格式
            symbol_df.to_parquet(symbol_file, index=False)

    def preprocess_data(self, stock_code, process_trade=False, process_order=False):
        """
        读入指定文件位置(stock_code)的.xz文件, 并进一步预处理。

        参数:
        stock_code (str): 股票代码或路径，用于指定需要处理的 .xz 文件。
        process_trade (bool): 是否处理并返回 trade 数据，默认为 False。
        process_order (bool): 是否处理并返回 order 数据，默认为 False。
        """
        logging.info(f"Preprocessing data for {stock_code}")

        # 从 .xz 文件读取数据
        depth_path = os.path.join(self.config.raw_data_path, "depth", stock_code)
        order_path = os.path.join(self.config.raw_data_path, "order", stock_code)
        trade_path = os.path.join(self.config.raw_data_path, "trade", stock_code)

        # 使用 utils.py 中的方法读取 capnp 数据
        depth_df = capnp_to_df(self.config, depth_path, "depth")

        if process_order:
            order_df = capnp_to_df(self.config, order_path, "order")
        if process_trade:
            trade_df = capnp_to_df(self.config, trade_path, "trade")

        # 处理 depth 数据
        depth_df["exchangeTsNanos"] = pd.to_datetime(depth_df["exchangeTsNanos"])


        # 处理 trade 数据
        if process_trade:
            trade_df['exchangeTs'] = pd.to_datetime(trade_df['exchangeTsNanos'])
            trade_df = trade_df[['symbolId', 'stockName', 'exchangeTsNanos', 'price', 'volume', 'orderId', 'tradeId', 'tradeType', 'weighted_price']]
            trade_df.set_index("exchangeTsNanos", inplace=True)
            self.trade = trade_df

        # 处理 order 数据
        if process_order:
            order_df['exchangeTs'] = pd.to_datetime(order_df['exchangeTsNanos'])
            order_df = order_df[['symbolId','stockName','exchangeTsNanos','price', 'volume', 'dir', 'orderId', 'newOrderId', 'updateType']]

            # 按 orderId 分组，提取价格非零的第一行
            filtered_order_df = order_df[order_df['price'] > 0].drop_duplicates(subset='orderId', keep='first')
            if process_trade:
                merged_df = pd.merge(trade_df, filtered_order_df[['price', 'orderId','dir']], on='orderId', how='left', suffixes=('', '_order'))
                print("合并后的 DataFrame 长度是否保持一致:", len(merged_df) == len(trade_df))
                trade_df = merged_df
            
            order_df.set_index("exchangeTsNanos", inplace=True)

            self.order = order_df

        # 处理 depth 数据
        depth_df['exchangeTs'] = pd.to_datetime(depth_df['exchangeTsNanos'])
        depth_df[["preClose", "open", "high", "low", "latestPrice"]] = depth_df[["preClose", "open", "high", "low", "latestPrice"]].replace(0, np.nan)
        depth_df["volume"] = depth_df["totalVolume"].diff(1)
        
        # 处理 10 档买卖量价
        depth_df['bid_prices'] = depth_df["bids"].apply(lambda x: np.array([i["price"] if i["price"] > 0 else np.nan for i in x]))
        depth_df['bid_volumes'] = depth_df["bids"].apply(lambda x: np.array([i["volume"] for i in x]))
        depth_df['bid_nums'] = depth_df["bids"].apply(lambda x: np.array([i["no"] for i in x]))

        depth_df['ask_prices'] = depth_df["asks"].apply(lambda x: np.array([i["price"] if i["price"] > 0 else np.nan for i in x]))
        depth_df['ask_volumes'] = depth_df["asks"].apply(lambda x: np.array([i["volume"] for i in x]))
        depth_df['ask_nums'] = depth_df["asks"].apply(lambda x: np.array([i["no"] for i in x]))
        # depth_df.drop(columns=['marketId', 'symbolType', 'status', 'spiderTsNanos',
        #     'depthTsNanos', 'totalBuyNo', 'totalSellNo', 'totalNo',
        #     'totalVolume', 'totalTurnover', 'cancelBuyNo', 'cancelBuyVolume',
        #     'cancelBuyTurnover', 'cancelSellNo', 'cancelSellVolume',
        #     'cancelSellTurnover','totalBidVolume', 'totalAskVolume', 'bidLevelsNum', 'askLevelsNum',
        #     'highLimited', 'lowLimited', 'eventByOrder', 'eventId', 'channelId', 'exchangeTs'], inplace = True)

        depth_df.set_index("exchangeTsNanos", inplace=True)
        self.depth = depth_df
    
    
