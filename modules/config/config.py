import capnp
import os

class BaseConfig:
    def __init__(self):
        # 通用配置
        self.backtest_output_folder = "./backtest_results"  # 回测结果存放路径
        self.start_date = "2024-01-01"  # 回测开始时间
        self.end_date = "2024-12-31"  # 回测结束时间
        self.initial_capital = 1000000  # 初始资本
        self.commission_rate = 0.0001  # 手续费
        self.slippage = 0.001  # 滑点

class HFTConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # HFT 数据配置
        self.raw_data_path = "./data/raw_data"  # 原始数据文件夹
        self.processed_data_path = "./data/processed_data"  # 处理后的数据文件夹
        self.factor_output_folder = "./generated_factors"  # 存放生成因子的文件夹
        self.solo_factor_output_folder = "./generated_factors/solo"
        self.assembled_factor_output_folder = "./generated_factors/assembled"
        self.generated_superfactors_path = "./generated_superfactors"
        self.factor_file_format = "parquet"  # 因子文件格式

        # # Cap'n Proto 配置
        # self.schema_path = '/home/lbz/BackTest/capnp/'
        # self.depth_capnp_file = "depthv3.capnp"
        # self.order_capnp_file = "order_u1.capnp"
        # self.trade_capnp_file = "trade_u1.capnp"
        # self.status_capnp_file = "status_u1.capnp"
        # self.struct_dict = {
        #     "depth": capnp.load(f'{self.schema_path}depthv3.capnp').DepthV3,
        #     "order": capnp.load(f'{self.schema_path}order_u1.capnp').OrderU1,
        #     "trade": capnp.load(f'{self.schema_path}trade_u1.capnp').TradeU1,
        #     "status": capnp.load(f'{self.schema_path}status_u1.capnp').StatusU1,
        # }

class DailyConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # 日频数据配置
        self.raw_data_path = "./stk_data-2020_2023/"
        self.daily_files = {
            "stk_daily": os.path.join(self.raw_data_path, "stk_daily-2020_2023.feather"),
            "stk_fin_annotation": os.path.join(self.raw_data_path, "stk_fin_annotation-2020_2023.feather"),
            "stk_fin_balance": os.path.join(self.raw_data_path, "stk_fin_balance-2020_2023.feather"),
            "stk_fin_cashflow": os.path.join(self.raw_data_path, "stk_fin_cashflow-2020_2023.feather"),
            "stk_fin_income": os.path.join(self.raw_data_path, "stk_fin_income-2020_2023.feather"),
            "stk_fin_item": os.path.join(self.raw_data_path, "stk_fin_item_map.feather"),
        }
