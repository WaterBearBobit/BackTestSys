import os
import re
import lzma
import pandas as pd
import numpy as np

def calculate_label(df, col_name, intervals):
    """
    Calculate returns for specified time intervals.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with bid/ask price columns.
        col_name (str): The column to calculate returns for.
        intervals (list of int): List of intervals in seconds.
    
    Returns:
        pd.DataFrame: A DataFrame with return columns for each interval.
    """
    labels = {}
    for interval in intervals:
        col_label = f"{col_name}_{interval}s"
        df[col_label] = df.index + pd.Timedelta(seconds=interval)
        labels[col_label] = df.apply(lambda row: _calculate_single_label(df, row, col_label), axis=1)
    return pd.concat(labels, axis=1)

def _calculate_single_label(df, row, col_label):
    """
    Helper function to calculate single label for a row.
    """
    window_label = df.loc[row.name:row[col_label]]
    if len(window_label) > 2:
        ask_1_first = window_label["ask_price_1"].values[0]
        bid_1_first = window_label["bid_price_1"].values[0]
        ask_1_last = window_label["ask_price_1"].values[-1]
        bid_1_last = window_label["bid_price_1"].values[-1]
        return ((ask_1_last + bid_1_last) - (ask_1_first + bid_1_first)) / (ask_1_first + bid_1_first)
    else:
        return np.nan

def compute_raw_features(df):
    """
    Compute raw features from bid/ask data for each rolling window.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with required bid/ask columns.
    
    Returns:
        pd.DataFrame: Aggregated features.
    """
    def aggregate_row(row):
        window = df.loc[row['lookback_start']:row.name]
        bid_price_1 = window['bid_price_1']
        ask_price_1 = window['ask_price_1']
        bid_vlm_1 = window['bid_vlm_1']
        ask_vlm_1 = window['ask_vlm_1']

        return pd.Series({
            'open': window['latestPrice'].iloc[0],
            'high': window['latestPrice'].max(),
            'low': window['latestPrice'].min(),
            'close': window['latestPrice'].iloc[-1],
            'volume': window['volume'].sum(),
            'first_bidPx1': bid_price_1.iloc[0],
            'first_askPx1': ask_price_1.iloc[0],
            'max_bidPx1': bid_price_1.max(),
            'max_askPx1': ask_price_1.max(),
            'min_bidPx1': bid_price_1.min(),
            'min_askPx1': ask_price_1.min(),
            'last_bidPx1': bid_price_1.iloc[-1],
            'last_askPx1': ask_price_1.iloc[-1],
            'last_bidVlm1': bid_vlm_1.iloc[-1],
            'last_askVlm1': ask_vlm_1.iloc[-1]
        })

    return df.apply(aggregate_row, axis=1)



def capnp_to_df(config, filename, dtype):
    """
    Convert a capnp file to a pandas DataFrame based on the specified dtype.

    Parameters:
        config: An object containing schema path and capnp file configurations.
        filename: The path to the binary capnp file.
        dtype: The type of data to process (e.g., "depth", "order", "trade", "status").

    Returns:
        pd.DataFrame: The processed data as a pandas DataFrame.
    """
    # Validate dtype
    if dtype not in config.struct_dict:
        raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: {list(config.struct_dict.keys())}")

    # Helper function to get fields order from a capnp schema file
    def get_fields_order(capnp_file):
        with open(capnp_file, 'r') as file:
            content = file.read()
        content = content.split("struct PriceVolume ")[0]

        # Regular expression to match field definitions
        field_pattern = re.compile(r'\s*(\w+)\s*@[0-9]+\s*:\s*(\S+)\s*;', re.MULTILINE)
        fields = field_pattern.findall(content)

        # Extract field names in order
        return [field[0] for field in fields]

    # Map dtype to capnp schema files and extract field orders
    schema_map = {
        "depth": config.schema_path + config.depth_capnp_file,
        "order": config.schema_path + config.order_capnp_file,
        "trade": config.schema_path + config.trade_capnp_file,
        "status": config.schema_path + config.status_capnp_file,
    }

    if dtype not in schema_map:
        raise ValueError(f"Schema file for dtype '{dtype}' is not configured.")

    # Get field names
    field_names = get_fields_order(schema_map[dtype])

    # Load and process the capnp binary file
    struct = config.struct_dict[dtype]
    f_in = lzma.decompress(open(filename, 'rb').read())
    messages = struct.read_multiple_bytes_packed(f_in)

    # Extract data into a list of lists
    data = []
    for message in messages:
        data.append([getattr(message, field) for field in field_names])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=field_names)

    # Post-processing based on dtype
    if dtype == "depth":
        df["bids"] = df["bids"].apply(lambda row: [{"price": x.price, "volume": x.volume, "no": x.no} for x in row])
        df["asks"] = df["asks"].apply(lambda row: [{"price": x.price, "volume": x.volume, "no": x.no} for x in row])
    else:
        df["stockName"] = df["stockName"].str.decode("utf-8")
        df["session"] = df["session"].str.decode("utf-8")

    return df


def load_first_parquet(folder_path):
    """
    从指定文件夹中加载第一个 .parquet 文件为 DataFrame。

    :param folder_path: str, 包含 .parquet 文件的文件夹路径。
    :return: DataFrame, 第一个 .parquet 文件的数据。
    """
    # 获取所有 .parquet 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {folder_path}")
    
    # 加载第一个文件
    first_file = os.path.join(folder_path, files[0])
    return pd.read_parquet(first_file)


def load_all_parquets(folder_path):
    """
    从指定文件夹中加载所有 .parquet 文件，并将它们拼接成一个长 DataFrame。

    :param folder_path: str, 包含 .parquet 文件的文件夹路径。
    :return: DataFrame, 拼接后的数据。
    """
    # 获取所有 .parquet 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {folder_path}")
    
    # 逐个加载文件并拼接
    dfs = [pd.read_parquet(os.path.join(folder_path, f)) for f in files]
    return pd.concat(dfs, axis=0).reset_index(drop=True)
