from modules.utils import load_first_parquet, load_all_parquets
from modules.config.config import HFTConfig, DailyConfig, BaseConfig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import pearsonr

def calculate_annualized_return(daily_returns: pd.Series):
    """计算年化收益"""
    return (1 + daily_returns.mean()) ** 252 - 1


def calculate_annualized_volatility(daily_returns: pd.Series):
    """计算年化波动"""
    return daily_returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate=0.03):
    """计算夏普比率（假设无风险利率为3%年化）"""
    excess_daily_returns = daily_returns - (risk_free_rate / 252)
    annualized_excess_return = calculate_annualized_return(excess_daily_returns)
    annualized_volatility = calculate_annualized_volatility(daily_returns)
    return annualized_excess_return / annualized_volatility if annualized_volatility > 0 else np.nan


def calculate_max_drawdown(cumulative_returns: pd.Series):
    """计算最大回撤"""
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    return drawdown.min()


def summarize_backtest_result_by_stock(result: pd.DataFrame):
    """
    按 stk_id 分类输出回测结果的摘要
    """
    summary = []
    
    # 计算股票累计收益率
    result['stock_cumulative_return'] = result.groupby('stk_id')['return'].transform(
        lambda x: (1 + x).cumprod() - 1
    )
    
    # 按 stk_id 分组计算指标
    grouped = result.groupby('stk_id')
    for stk_id, group in grouped:
        strategy_returns = group['strategy_return']
        stock_returns = group['return']
        cumulative_returns = group['cumulative_return']
        
        summary.append({
            'stk_id': stk_id,
            'excess_return': (strategy_returns.mean() - stock_returns.mean()) * 252,
            'annualized_return': calculate_annualized_return(strategy_returns),
            'annualized_volatility': calculate_annualized_volatility(strategy_returns),
            'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
            'max_drawdown': calculate_max_drawdown(cumulative_returns)
        })
    
    return pd.DataFrame(summary)


def summarize_backtest_result_average(result: pd.DataFrame):
    """
    汇总所有 stk_id 的结果，计算均值
    """
    # 计算股票累计收益率
    result['stock_cumulative_return'] = result.groupby('stk_id')['return'].transform(
        lambda x: (1 + x).cumprod() - 1
    )
    
    # 按 stk_id 计算各指标
    stk_summary = summarize_backtest_result_by_stock(result)

    # 汇总所有 stk_id 的均值
    avg_summary = {
        'excess_return': stk_summary['excess_return'].mean(),
        'annualized_return': stk_summary['annualized_return'].mean(),
        'annualized_volatility': stk_summary['annualized_volatility'].mean(),
        'sharpe_ratio': stk_summary['sharpe_ratio'].mean(),
        'max_drawdown': stk_summary['max_drawdown'].mean()
    }
    
    return avg_summary

def plot_daily_and_cumulative_returns(data, stk_id):
    # 过滤出指定股票的数据
    data['date'] = pd.to_datetime(data['date'])
    stock_data = data[data['stk_id'] == stk_id].copy()

    # 计算策略的累计收益率
    stock_data['strategy_return'] = stock_data['strategy_return'].fillna(0)
    stock_data['cumulative_strategy_return'] = (1 + stock_data['strategy_return']).cumprod() - 1

    # 计算基准（股票本身的）累计收益率
    stock_data['return'] = stock_data['return'].fillna(0)
    stock_data['cumulative_return'] = (1 + stock_data['return']).cumprod() - 1

    # 绘制图像
    plt.figure(figsize=(12, 6))

    # 日收益图
    plt.subplot(2, 1, 1)
    plt.plot(stock_data['date'], stock_data['return'], label='Stock Daily Return', alpha=0.6)
    plt.plot(stock_data['date'], stock_data['strategy_return'], label='Strategy Daily Return', alpha=0.6)
    plt.title(f"Daily Return of Stock {stk_id}")
    plt.legend()

    # 累计收益图
    plt.subplot(2, 1, 2)
    plt.plot(stock_data['date'], stock_data['cumulative_return'], label='Stock Cumulative Return', alpha=0.6)
    plt.plot(stock_data['date'], stock_data['cumulative_strategy_return'], label='Strategy Cumulative Return', alpha=0.6)
    plt.title(f"Cumulative Return of Stock {stk_id}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_overall_strategy_return(data):
    # 按日期聚合所有股票的日收益
    data['strategy_return'] = data['strategy_return'].fillna(0)
    data['return'] = data['return'].fillna(0)
    
    daily_strategy_return = data.groupby('date')['strategy_return'].mean()
    daily_stock_return = data.groupby('date')['return'].mean()

    # 计算累计收益
    cumulative_strategy_return = (1 + daily_strategy_return).cumprod() - 1
    cumulative_stock_return = (1 + daily_stock_return).cumprod() - 1

    # 绘制图像
    plt.figure(figsize=(12, 12))

    # 日收益图
    plt.subplot(2, 1, 1)
    plt.plot(daily_strategy_return.index, daily_strategy_return.values, label='Strategy Daily Return')
    plt.plot(daily_stock_return.index, daily_stock_return.values, label='Stock Daily Return', linestyle='--')
    plt.title('Daily Return Comparison')
    plt.legend()

    # 累计收益图
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_strategy_return.index, cumulative_strategy_return.values, label='Strategy Cumulative Return')
    plt.plot(cumulative_stock_return.index, cumulative_stock_return.values, label='Stock Cumulative Return', linestyle='--')
    plt.title('Cumulative Return Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_factor_statistics(data):
    # 计算排名IC和Pearson相关系数
    data['date'] = pd.to_datetime(data['date'])
    data['factor'] = data['factor'].fillna(0)  # 填充缺失值
    data['return'] = data['return'].fillna(0)  # 填充缺失值
    daily_ic = []
    daily_pearson = []

    for date, group in data.groupby('date'):
        factor = group['factor']
        signal = group['return']

        # 计算因子IC (Rank Correlation)
        ic = factor.corr(signal, method='spearman')
        daily_ic.append(ic)

        # 计算Pearson相关系数
        pearson_corr, _ = pearsonr(factor, signal)
        daily_pearson.append(pearson_corr)

    # 绘制IC和Pearson系数图像
    plt.figure(figsize=(12, 6))
    plt.plot(list(data['date'].unique()), daily_ic, label='Rank IC', alpha=0.6)
    plt.plot(list(data['date'].unique()), daily_pearson, label='Pearson Correlation', alpha=0.6)
    plt.title('Factor Statistics (Rank IC & Pearson Correlation)')
    plt.legend()
    plt.show()

def generate_visualizations(data):
    # 生成指定股票的日收益和累计收益图
    # plot_daily_and_cumulative_returns(data, stk_id)

    # 生成所有股票的综合策略收益图
    plot_overall_strategy_return(data)

    # 生成因子统计图
    plot_factor_statistics(data)


def hft_generate_heatmap():
    """
    生成因子相关性的热力图
    """
    config = HFTConfig()
    factor_df = load_first_parquet(config.solo_factor_output_folder)  # 加载单因子

    # 保留因子列
    factor_data = factor_df.iloc[:, 3:]

    # 计算相关性矩阵
    corr_matrix = factor_data.corr()

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0)
    plt.title("Factor Correlation Heatmap")
    plt.show()

def hft_plot_pnl(backtest_result_file):
    """
    绘制 PnL 图像（支持多个 PnL 列），并以 latestPrice 作为基准
    """
    config = HFTConfig()
    pnl_df = pd.read_parquet(f"{config.backtest_output_folder}/{backtest_result_file}")
    
    if 'latestPrice' not in pnl_df.columns:
        raise ValueError("Backtest result file must contain 'latestPrice' column.")

    pnl_columns = [col for col in pnl_df.columns if re.search(r'pnl$', col)]
    if not pnl_columns:
        raise ValueError("No columns ending with 'pnl' were found in the backtest result file.")
    
    pnl_df['cumulative_price'] = pnl_df['latestPrice'] / pnl_df['latestPrice'].iloc[0] - 1  # 基准收益率

    plt.figure(figsize=(12, 6))

    for pnl_col in pnl_columns:
        pnl_df[f'cumulative_{pnl_col}'] = pnl_df[pnl_col].cumsum()
        plt.plot(pnl_df['exchangeTsNanos'], pnl_df[f'cumulative_{pnl_col}'], label=f"Cumulative {pnl_col}")
    
    plt.plot(pnl_df['exchangeTsNanos'], pnl_df['cumulative_price'], label="Benchmark (latestPrice)", color='orange', linestyle='--')
    
    # 图像格式设置
    plt.title("PnL vs Benchmark")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()