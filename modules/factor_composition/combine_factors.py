# factor_composition/combine_factors.py

import pandas as pd

class FactorComposition:
    def __init__(self, config):
        self.config = config
    
    def combine_factors(self, factor_df):
        """
        组合基础因子形成超级因子（这里只是简单示范，可以使用更复杂的模型）
        """
        # 假设超级因子是多个基础因子的加权和
        factor_df['super_factor'] = factor_df['price_change'] * 0.5  # 示例：用一个因子

        # 返回新的DataFrame，包含超级因子
        return factor_df[['symbolId', 'nanoTime', 'super_factor']]
    
    def save_super_factors(self, super_factor_df):
        """
        保存超级因子为parquet格式
        """
        super_factor_df.to_parquet(self.config.generated_superfactors_path)
        print(f"Super factors saved to {self.config.generated_superfactors_path}")
