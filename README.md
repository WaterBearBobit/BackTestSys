# README

---

### **模块结构**

系统设计基于模块化架构，各模块职责明确、功能独立，便于扩展和维护。核心模块设计如下：  
```
project
├── backtest_results/    # 存放回测结果的目录
├── data/                # 存放数据的目录
├── generated_factors/   # 存放生成的因子数据的目录
├── modules/             # 存放回测系统模块的目录
│   ├── __pycache__      # 存放Python编译后的字节码文件
│   ├── backtest/        # 回测模块
│   │   ├── __pycache__  # 存放回测模块编译后的字节码文件
│   │   ├── backtrader.py  # 基于Backtrader的回测引擎
│   │   └── strategy.py    # 交易策略定义
│   ├── config/          # 配置模块
│   │   ├── __pycache__  # 存放配置模块编译后的字节码文件
│   │   └── config.py    # 配置管理脚本
│   ├── factor_composition/  # 因子组合模块
│   │   ├── __pycache__      # 存放因子组合模块编译后的字节码文件
│   │   └── combine_factors.py  # 因子组合逻辑实现
│   └── factor_construction/   # 因子构造模块
│       ├── __pycache__        # 存放因子构造模块编译后的字节码文件
│       ├── compute_engine.py  # 计算引擎，用于因子计算
│       ├── data_engine.py     # 数据引擎，用于数据获取和处理
│       ├── __init__.py        # 初始化模块，可能用于模块级别的变量或函数
│       ├── utils.py           # 工具函数，提供辅助功能
│       └── visualization.py   # 结果可视化模块，用于展示因子构造结果
├── stk_data-2020_2023  # 存放股票数据的文件
└── main.ipynb          # 主程序入口，Jupyter Notebook格式
```

---

以下是系统数据流的结构化展示图：  

```plaintext
               ┌──────────────────┐
    Data       │  data_engine.py  │  Processed Data
 ────────────► │                  │ ────────────────►
               └──────────────────┘
                     │
                     ▼
 ┌──────────────────────┐     ┌──────────────────────┐
 │ compute_engine.py    │ ──► │ combine_factors.py   │───►  Factor Results
 └──────────────────────┘     └──────────────────────┘
                     │
                     ▼
 ┌──────────────┐     ┌──────────────┐
 │ backtrader.py│ ──► │ strategy.py  │───►  Strategy Results
 └──────────────┘     └──────────────┘
```

---
# 回测系统概述

本回测系统是一个综合性的框架，用于开发、测试和评估交易策略。系统提供了从数据加载、因子计算、策略回测到结果可视化的全套工具。以下是系统的主要功能模块和它们的作用：

## 1. 配置模块 (config.py)
定义了回测所需的基础配置，包括回测时间范围、初始资本、手续费率和滑点等。同时，为高频和日频交易提供了特定的配置。

## 2. 数据引擎模块 (data_engine.py)
提供了数据加载和预处理的功能，支持高频和日频数据。用户可以根据数据频率获取相应的数据引擎实例，进行数据的加载和清洗。

## 3. 因子构造模块 (factor_construction.py)
包含了一系列用于计算股票因子的类，如日频因子、财务注释因子、财务平衡表因子、财务现金流因子和财务收入表因子。这些因子可以用于策略开发和回测。

## 4. 因子组合模块 (combine_factors.py)
提供了将基础因子组合成超级因子的功能，允许用户通过加权或其他模型来创建新的交易信号。

## 5. 策略模块 (strategy.py)
定义了交易策略的类，用户可以根据不同的策略生成买卖信号。系统支持自定义策略，并提供了基于形态识别和因子的策略示例。

## 6. 回测引擎模块 (backtest_engine.py)
实现了回测的核心功能，包括策略收益率的计算、手续费和滑点的考虑，以及累积收益率的计算。提供了多种回测方法，以适应不同的交易策略和需求。

## 7. 工具模块 (utils.py)
提供了一组实用函数，用于辅助数据处理和回测引擎的工作，如计算收益率标签、加载 parquet 文件、将 capnp 文件转换为 DataFrame 等。

## 8. 可视化模块 (visualization.py)
提供了数据可视化的功能，用于展示回测结果和因子统计信息。用户可以生成日收益和累计收益图、因子相关性热力图、PnL 图像等，以直观地理解策略性能和因子特性。

## 使用示例
用户可以通过以下步骤使用本回测系统：
1. 配置回测参数。
2. 加载和预处理数据。
3. 计算和组合交易因子。
4. 定义和执行交易策略。
5. 运行回测并获取结果。
6. 可视化回测结果和因子统计。

这个回测系统旨在为交易策略的开发和评估提供一个全面、灵活和易于使用的工具集。

--- 

# 模块：factor_construction/data_engine.py

该模块提供了数据引擎的抽象基类和具体实现，用于处理高频（HFT）和低频（Daily）数据。

## 类结构概述

### DataEngineFactory
数据引擎工厂类，用于根据数据频率创建相应的数据引擎实例。

**属性：**
- 无

**方法：**
- `get_data_engine(data_frequency, raw_data_path)`: 根据数据频率和原始数据路径返回数据引擎实例。

### DataEngine (ABC)
数据引擎的抽象基类，定义了数据加载、清洗和处理的通用接口。

**属性：**
- `raw_data_path`: 原始数据路径。
- `processed_data`: 处理后的数据。

**方法：**
- `load_data()`: 加载原始数据。
- `clean_data()`: 清洗数据。
- `process_frequency_specific_data()`: 子类必须实现的频率特定数据处理方法。
- `get_processed_data()`: 获取处理后的数据。

### DailyDataEngine
日频数据引擎，继承自 `DataEngine`，实现日频数据的加载和预处理。

**属性：**
- `config`: 日频数据配置。
- `data`: 加载的数据字典。

**方法：**
- `load_data()`: 加载所有日频数据文件。
- `preprocess_data()`: 对加载的数据进行预处理。
- `get_data(key)`: 根据键获取特定的数据。

### HFTDataEngine
高频数据引擎，继承自 `DataEngine`，实现高频数据的加载和预处理。

**属性：**
- `config`: 高频数据配置。

**方法：**
- `split_data()`: 处理并分割parquet数据。
- `preprocess_data(stock_code, process_trade=False, process_order=False)`: 读入指定文件位置的.xz文件，并进行预处理。

## 示例用法

### 创建数据引擎实例
```markdown
# 创建日频数据引擎实例
daily_engine = DataEngineFactory.get_data_engine('low', '/path/to/daily/data')

# 创建高频数据引擎实例
hft_engine = DataEngineFactory.get_data_engine('high', '/path/to/hft/data')

--- 
# 模块：config.py

该模块定义了回测系统的基础配置类，以及针对高频（HFT）和日频（Daily）数据的特定配置。

## 类结构概述

### BaseConfig
基础配置类，包含所有回测配置的通用属性。

**属性：**
- `backtest_output_folder`: 回测结果存放路径。
- `start_date`: 回测开始时间。
- `end_date`: 回测结束时间。
- `initial_capital`: 初始资本。
- `commission_rate`: 手续费率。
- `slippage`: 滑点。

### HFTConfig (继承自 BaseConfig)
高频数据配置类，包含高频数据特定的属性。

**属性：**
- `raw_data_path`: 原始高频数据文件夹。
- `processed_data_path`: 处理后的高频数据文件夹。
- `factor_output_folder`: 存放生成因子的文件夹。
- `solo_factor_output_folder`: 存放单独因子的文件夹。
- `assembled_factor_output_folder`: 存放组合因子的文件夹。
- `generated_superfactors_path`: 超级因子的保存路径。
- `factor_file_format`: 因子文件格式。

### DailyConfig (继承自 BaseConfig)
日频数据配置类，包含日频数据特定的属性。

**属性：**
- `raw_data_path`: 原始日频数据文件夹。
- `daily_files`: 包含日频数据文件路径的字典。

## 示例用法

### 初始化配置
```markdown
# 初始化高频配置
hft_config = HFTConfig()

# 初始化日频配置
daily_config = DailyConfig()
---

# 模块：factor_construction

该模块包含了一系列用于计算股票因子的类，包括日频因子、财务注释因子、财务平衡表因子、财务现金流因子和财务收入表因子。此外，还包括用于计算和存储因子的引擎类。

## 类结构概述

### StkDailyFactors
股票日频因子计算类。

**属性：**
- `data`: 股票日频数据 DataFrame。

**方法：**
- `compute_factors_for_group(group)`: 对单个股票分组计算因子。
- `get_factors()`: 计算并返回包含所有计算后的因子的 DataFrame。

### StkFinAnnotationFactors
财务注释因子计算类。

**属性：**
- `data`: 股票财务注释数据 DataFrame。

**方法：**
- `compute_nonrecurring_total()`: 计算非经常性损益总和。
- `get_factors()`: 计算并返回包含计算后的财务注释因子的 DataFrame。

### StkFinBalanceFactors
财务平衡表因子计算类。

**属性：**
- `data`: 股票财务平衡表数据 DataFrame。

**方法：**
- `compute_debt_to_asset()`: 计算资产负债比。
- `get_factors()`: 计算并返回包含计算后的财务平衡表因子的 DataFrame。

### StkFinCashflowFactors
财务现金流因子计算类。

**属性：**
- `data`: 股票财务现金流数据 DataFrame。

**方法：**
- `compute_operating_cashflow_ratio()`: 计算经营活动现金流占比。
- `get_factors()`: 计算并返回包含计算后的财务现金流因子的 DataFrame。

### StkFinIncomeFactors
财务收入表因子计算类。

**属性：**
- `data`: 股票财务收入表数据 DataFrame。

**方法：**
- `compute_net_profit_margin()`: 计算净利润率。
- `get_factors()`: 计算并返回包含计算后的财务收入表因子的 DataFrame。

### DailyComputeEngine
日频计算引擎类，根据数据类型选择相应的因子类进行计算。

**属性：**
- `data`: 数据 DataFrame。
- `data_type`: 数据类型。
- `factors`: 根据数据类型选择的因子类实例。

**方法：**
- `compute_factors()`: 计算因子，返回计算结果。

### HFTComputeEngine
高频计算引擎类，用于处理高频数据并计算因子。

**属性：**
- `depth`: 深度数据 DataFrame。
- `order`: 订单数据 DataFrame。
- `trade`: 交易数据 DataFrame。
- `factors`: 存储计算后的因子。
- `rolling_sec`: 滚动窗口秒数。

**方法：**
- `pre_process_depth()`: 预处理深度数据。
- `calculate_returns(intervals)`: 计算给定间隔的回报。
- `calculate_raw_features()`: 计算原始特征。
- `S1001()` 到 `S1011()`: 计算特定的高频因子。
- `calculate_IC()`: 计算信息系数 (IC)。
- `save_factors(factor_df)`: 将计算的因子数据保存为 parquet 格式。

## 示例用法

### 计算日频因子
```markdown
daily_data = pd.read_feather('path_to_daily_data.feather')
daily_factors_engine = DailyComputeEngine(daily_data, data_type='stk_daily')
daily_factors = daily_factors_engine.compute_factors()
---
# 模块：factor_composition/combine_factors.py

该模块提供了一个用于组合基础因子形成超级因子的类。超级因子可以是基础因子的加权和或其他更复杂的模型。

## 类结构概述

### FactorComposition
因子组合类，用于将基础因子组合成超级因子。

**属性：**
- `config`: 配置对象，包含生成超级因子的路径等信息。

**方法：**
- `combine_factors(factor_df)`: 组合基础因子形成超级因子。
- `save_super_factors(super_factor_df)`: 保存超级因子为 parquet 格式。

## 示例用法

### 组合因子
```markdown
# 假设 factor_df 是一个包含多个基础因子的 DataFrame
factor_composer = FactorComposition(config)
super_factor_df = factor_composer.combine_factors(factor_df)
---
# 模块：strategy.py

该模块定义了用于生成买卖信号的策略类，包括基于特定形态识别和因子的策略。

## 类结构概述

### Reversal
基于反转因子的策略类。

**方法：**
- `buy_sell_points()`: 根据反转因子生成买卖信号。

### Bearish_Engulfing
基于看跌吞没形态的策略类。

**方法：**
- `strategy_bearish_engulfing(group, idx, parameter)`: 判断是否满足看跌吞没形态并返回信号。
- `buy_sell_points()`: 根据看跌吞没形态策略生成买卖信号。

### Strategy
策略类，根据指定的策略生成买卖信号。

**属性：**
- `data`: 包含股票数据的 DataFrame。
- `strategy`: 策略对象，可以是 `Reversal` 或 `Bearish_Engulfing`。

**方法：**
- `generate_signals()`: 根据指定策略生成买卖信号。

### HFTStrategy
高频交易策略类。

**属性：**
- `config`: 高频交易配置对象。

**方法：**
- `generate_signals(factor_df)`: 基于因子生成买卖信号。
- `generate_super_factor_signals(super_factor_df)`: 基于超级因子生成买卖信号（待实现）。

## 示例用法

### 初始化策略并生成信号
```markdown
# 初始化反转策略并生成信号
reversal_strategy = Strategy(data, strategy='Reversal')
signals = reversal_strategy.generate_signals()

# 初始化看跌吞没策略并生成信号
bearish_engulfing_strategy = Strategy(data, strategy='Bearish_Engulfing')
signals = bearish_engulfing_strategy.generate_signals()
---
# 模块：backtest_engine.py

该模块提供了回测引擎的实现，用于评估交易策略的性能。它包括基础回测引擎和针对高频交易（HFT）的回测引擎。

## 类结构概述

### BacktestEngine
基础回测引擎类，用于评估交易策略的性能。

**属性：**
- `df`: 包含交易数据的 DataFrame。
- `config`: 回测参数的配置类实例。

**方法：**
- `set_date(start_date, end_date)`: 设置回测的开始和结束日期。
- `calculate_strategy_return(consider_commission, consider_slippage)`: 计算策略的收益率，考虑手续费和滑点。
- `backtest(consider_commission, consider_slippage)`: 执行回测，计算策略的累积收益率。
- `backtest_no_commission()`: 执行回测，不考虑手续费。
- `backtest_no_slippage()`: 执行回测，不考虑滑点。
- `backtest_no_commission_no_slippage()`: 执行回测，不考虑手续费和滑点。

### HFTBacktestEngine
高频交易回测引擎类，用于评估高频交易策略的性能。

**属性：**
- `config`: 高频交易配置对象。
- `strategy`: 交易策略对象。
- `portfolio`: 投资组合，包括现金和持仓。
- `orders`: 交易订单记录。

**方法：**
- `execute_trades(signals_df)`: 根据信号执行交易操作，更新投资组合和订单记录。
- `run_backtest()`: 运行回测，加载因子文件，生成信号，并执行交易。
- `run_assembled_backtest()`: 运行组合因子的回测（待实现）。

## 示例用法

### 初始化回测引擎并执行回测
```markdown
# 初始化基础回测引擎
backtest_engine = BacktestEngine(data)
backtest_engine.set_date('2024-01-01', '2024-12-31')
results = backtest_engine.backtest()

# 初始化高频交易回测引擎
hft_backtest_engine = HFTBacktestEngine(strategy)
pnl_over_time = hft_backtest_engine.run_backtest()
---
# 模块：utils.py

该模块提供了一组实用函数，用于辅助数据处理和回测引擎的工作。

## 函数列表

### calculate_label
计算指定时间间隔的收益率标签。

**参数：**
- `df (pd.DataFrame)`: 输入的 DataFrame，包含买卖价格列。
- `col_name (str)`: 计算收益率的列名。
- `intervals (list of int)`: 时间间隔列表，以秒为单位。

**返回：**
- `pd.DataFrame`: 包含每个时间间隔收益率列的 DataFrame。

### _calculate_single_label
辅助函数，用于计算单行的收益率标签。

### compute_raw_features
从买卖数据中计算每个滚动窗口的原始特征。

**参数：**
- `df (pd.DataFrame)`: 输入的 DataFrame，包含所需的买卖列。

**返回：**
- `pd.DataFrame`: 聚合后的特征。

### capnp_to_df
将 capnp 文件转换为 pandas DataFrame，基于指定的数据类型。

**参数：**
- `config`: 包含 schema 路径和 capnp 文件配置的对象。
- `filename`: 二进制 capnp 文件的路径。
- `dtype`: 要处理的数据类型（例如："depth", "order", "trade", "status"）。

**返回：**
- `pd.DataFrame`: 处理后的数据作为 pandas DataFrame。

### load_first_parquet
从指定文件夹中加载第一个 `.parquet` 文件为 DataFrame。

**参数：**
- `folder_path (str)`: 包含 `.parquet` 文件的文件夹路径。

**返回：**
- `pd.DataFrame`: 第一个 `.parquet` 文件的数据。

### load_all_parquets
从指定文件夹中加载所有 `.parquet` 文件，并将它们拼接成一个长 DataFrame。

**参数：**
- `folder_path (str)`: 包含 `.parquet` 文件的文件夹路径。

**返回：**
- `pd.DataFrame`: 拼接后的数据。

## 示例用法

### 计算收益率标签
```markdown
# 假设 df 是包含 'bid_price_1' 和 'ask_price_1' 列的 DataFrame
intervals = [15, 30, 60]  # 以秒为单位的时间间隔
label_df = calculate_label(df, 'price', intervals)
---
## **Appendix**

### **计划应用LLM协助完成如下工作**

- **代码填充与效率优化：** 在已经实现的代码框架的基础上填充函数内容，并给出提高代码效率的建议。  
- **策略生成与系统测试：** 通过LLM生成新的因子逻辑或改进现有因子，生成多样的因子/策略来测试回测系统。  
- **文档与注释生成：** 根据回测系统代码和结果自动生成技术文档，或者在代码逻辑的关键部分补充大段注释。
