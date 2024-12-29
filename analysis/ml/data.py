import pandas as pd
import duckdb
import os

from numba import njit
from utils.db_utils import QUERY

@njit
def process_bars(dollar_values, quantities, dollar_threshold, open_prices, high_prices, low_prices, close_prices):
    bars = []
    cumulative_dollar_value = 0
    bar_quantity = 0
    bar_open = 0
    bar_high = 0
    bar_low = 0
    bar_close = 0
    bar_start_idx = 0

    for i in range(len(dollar_values)):
        if bar_quantity == 0:
            bar_open = open_prices[i]
            bar_low = low_prices[i]
            bar_high = high_prices[i]
        else:
            bar_low = min(bar_low, low_prices[i])
            bar_high = max(bar_high, high_prices[i])

        cumulative_dollar_value += dollar_values[i]
        bar_quantity += quantities[i]
        bar_close = close_prices[i]

        if cumulative_dollar_value >= dollar_threshold:
            bars.append((bar_start_idx, i, bar_open, bar_high, bar_low, bar_close, bar_quantity))
            cumulative_dollar_value = 0
            bar_quantity = 0
            bar_start_idx = i + 1

    return bars, cumulative_dollar_value, bar_quantity, bar_start_idx, bar_open, bar_high, bar_low, bar_close

def calculate_dollar_bars(token_symbol, dollar_threshold):
    """
    Calculates dollar bars for a given token based on the 1-minute OHLCV data.

    Args:
    - token_symbol (str): The symbol of the token to calculate dollar bars for.
    - dollar_threshold (float): The dollar value threshold for constructing each bar.

    Returns:
    - A pandas DataFrame containing the dollar bars.
    """   
    base, quote, exchange = token_symbol.split('_')

    # Query to select OHLCV data for the given token
    query = f'''
    SELECT origin_time, open, high, low, close, volume
    FROM market_data.ohlcv_1m
    WHERE 
        asset_id_base = '{base}' AND 
        asset_id_quote = '{quote}' AND
        exchange_id = '{exchange}'
    ORDER BY origin_time
    '''

    df = QUERY(query).fillna(0)
    
    # Use vectorized operations to calculate dollar values
    df['dollar_value'] = df['close'] * df['volume']

    dollar_values = df['dollar_value'].values
    quantities = df['volume'].values
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values

    # Use Numba-accelerated function to process bars
    result, cumulative_dollar_value, bar_quantity, bar_start_idx, bar_open, bar_high, bar_low, bar_close = process_bars(
        dollar_values, quantities, dollar_threshold, open_prices, high_prices, low_prices, close_prices
    )


    # Create bars DataFrame
    bars = []
    for bar_start, bar_end, open_price, high_price, low_price, close_price, quantity in result:
        bars.append({
            'origin_time': df['origin_time'].iloc[bar_start],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': quantity,
            'asset_id_base': base,
            'asset_id_quote': quote,
            'exchange_id': exchange
        })

    # Check if there is an incomplete bar after processing all data
    if cumulative_dollar_value > 0:
        bars.append({
            'origin_time': df['origin_time'].iloc[bar_start_idx],
            'open': bar_open,
            'high': bar_high,
            'low': bar_low,
            'close': bar_close,
            'volume': bar_quantity
        })

    # Return the bars as a pandas DataFrame
    return pd.DataFrame(bars)

def calculate_label_uniquness(X):
    symbol_ids = X['symbol_id'].unique()
    X['avg_uniqueness'] = 0
    for symbol_id in symbol_ids:
        # Sort the data by time period end and fill in missing dates in the index
        X_symbol = X[X['symbol_id'] == symbol_id].reset_index(drop = True)
        
        # Calculate label concurrency at each time period end
        start_date_index = X_symbol['start_date_triple_barrier_label_h7'].min()
        end_date_index = X_symbol['end_date_triple_barrier_label_h7'].max()
        label_concurrency = pd.DataFrame(index = pd.date_range(start_date_index, end_date_index, freq = 'D'))
        label_concurrency['label_concurrency'] = 0

        for i, (t0, t1) in X_symbol[['start_date_triple_barrier_label_h7', 'end_date_triple_barrier_label_h7']].iterrows():
            t0 = pd.Timestamp(t0)
            t1 = pd.Timestamp(t1)
            label_concurrency.loc[t0:t1, 'label_concurrency'] += 1

        # Uniqueness of label i at time t
        u_i_t = 1 / label_concurrency['label_concurrency']

        # The average uniqueness of label i is the mean of u_i_t over its lifespan
        for i, (t0, t1) in X_symbol[['start_date_triple_barrier_label_h7', 'end_date_triple_barrier_label_h7']].iterrows():
            t0 = pd.Timestamp(t0)
            t1 = pd.Timestamp(t1)
            u_i = u_i_t.loc[t0:t1].mean()
            X_symbol.loc[i, 'avg_uniqueness'] = u_i

        X.loc[X['symbol_id'] == symbol_id, 'avg_uniqueness'] = X_symbol['avg_uniqueness']

    return X
    
def construct_dataset_for_ml(resample_period):
    
    with duckdb.connect(
        database = '/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/database.db',
        read_only = False
    ) as conn:
        
        # Get all distinct assets in the database
        assets = conn.sql(
            """
            SELECT DISTINCT 
                asset_id_base, 
                asset_id_quote, 
                exchange_id 
            FROM market_data.ohlcv_1m 
            ORDER BY asset_id_base
        """).df()

        # Create an empty DataFrame to store the final dataset
        dataset = []

        # Keep track of the number of assets skipped due to missing data
        total_skipped = 0

        # Loop through each asset
        for i in range(len(assets)):
            print(f"Processing asset {i + 1} of {len(assets)} ({assets.iloc[i]['asset_id_base']}/{assets.iloc[i]['asset_id_quote']} on {assets.iloc[i]['exchange_id']})...")
            # Get the asset
            asset = assets.iloc[i]

            # Get the asset data
            data = conn.sql(
                f"""
                SELECT 
                    origin_time AS time_period_end,
                    asset_id_base,
                    asset_id_quote,
                    exchange_id,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    trades
                FROM market_data.ohlcv_1m
                WHERE
                    asset_id_base = '{asset['asset_id_base']}' AND
                    asset_id_quote = '{asset['asset_id_quote']}' AND
                    exchange_id = '{asset['exchange_id']}'
                ORDER BY origin_time
                """
            ).df().set_index('time_period_end').resample('1min', label = 'right', closed = 'left').agg({
                'asset_id_base': 'last',
                'asset_id_quote': 'last',
                'exchange_id': 'last',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trades': 'sum'
            })

            # Skip over tokens with more than 25% missing data
            pct_missing_price_close = data.loc[:,'close'].isna().mean() * 100
                        
            # Interpolate missing values
            numeric_cols = [col for col in data.columns if col not in ('asset_id_base', 'asset_id_quote', 'exchange_id')]
            categorical_cols = ['asset_id_base', 'asset_id_quote', 'exchange_id']

            data.loc[:,numeric_cols] = data.loc[:,numeric_cols].interpolate(method = 'time')

            for col in categorical_cols:
                mode = data.loc[:,col].mode().iloc[0]
                data.loc[:,col] = data.loc[:,col].fillna(mode)
                            
            # Downsample to resample_period
            data = data.resample(resample_period, label = 'right', closed = 'left').agg({
                'asset_id_base': 'last',
                'asset_id_quote': 'last',
                'exchange_id': 'last',
                'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'volume': 'sum',
                'trades': 'sum'
            })
            
            dataset.append(data)

        dataset = pd.concat(dataset)
        dataset = dataset.reset_index()
        dataset['time_period_end'] = pd.to_datetime(dataset['time_period_end'])

        print(f"Total assets skipped due to missing data: {total_skipped}")
        print(f'Percentage of assets skipped: {total_skipped / len(assets) * 100:.2f}%')

        # Save if the dataset file does not exist
        if not os.path.exists('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv'):
            dataset.to_csv('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv.gz', index = False)

            QUERY(
                """
                CREATE OR REPLACE TABLE market_data.ml_dataset as from read_csv('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv.gz');
                """
            )

        # Return the dataset
        return dataset

def get_ml_dataset(use_dollar_bars = False, resample_period = '1d'):
    # Construct the dataset for machine learning
    if not os.path.exists('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv.gz'):
        df = construct_dataset_for_ml(resample_period = resample_period)
        df['time_period_end'] = pd.to_datetime(df['time_period_end'])

        if use_dollar_bars:
            print('Calculating dollar bars...')
            dollar_bars_df = []
            symbols = df['asset_id_base'] + '_' + df['asset_id_quote'] + '_' + df['exchange_id']

            for symbol in symbols.unique():                
                dollar_bars_df.append(calculate_dollar_bars(symbol, 10_000_000))

            dollar_bars_df = pd.concat(dollar_bars_df)
            dollar_bars_df = dollar_bars_df.rename(columns = {'origin_time': 'time_period_end'})
            dollar_bars_df['time_period_end'] = pd.to_datetime(dollar_bars_df['time_period_end'])
            dollar_bars_df.to_csv('/Users/louisspencer/Desktop/Trading-Bot/data/dollar_bars.csv')

            return dollar_bars_df

        print('dataset shape:', df.shape)
        return df
    else:
        if use_dollar_bars:
            df = pd.read_csv('/Users/louisspencer/Desktop/Trading-Bot/data/dollar_bars.csv')
            df['time_period_end'] = pd.to_datetime(df['time_period_end'])
            return df
        else:
            df = pd.read_csv('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv.gz')
            df['time_period_end'] = pd.to_datetime(df['time_period_end'])
            return df

def get_ml_features():
    tokens = QUERY(
        """
        SELECT DISTINCT asset_id_base, asset_id_quote, exchange_id
        FROM market_data.ohlcv_1m
        ORDER BY asset_id_base, asset_id_quote, exchange_id
        """
    )

    if not os.path.exists('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.csv.gz'):
        QUERY(
            """
            TRUNCATE TABLE market_data.ml_features
            """
        )
        i = 1
        n = len(tokens[['asset_id_base', 'asset_id_quote', 'exchange_id']])

        for idx, row in tokens[['asset_id_base', 'asset_id_quote', 'exchange_id']].iterrows():
            asset_id_base = row['asset_id_base']
            asset_id_quote = row['asset_id_quote']
            exchange_id = row['exchange_id']
            symbol_id = f'{asset_id_base}_{asset_id_quote}_{exchange_id}'  

            print(f'Processing symbol_id: {symbol_id} ({i}/{n})')
            i += 1

            token = QUERY(
                f"""
                SELECT 
                    asset_id_base || '_' || asset_id_quote || '_' || exchange_id AS symbol_id,
                    *
                FROM market_data.ml_dataset
                WHERE
                    asset_id_base = '{asset_id_base}' AND
                    asset_id_quote = '{asset_id_quote}' AND
                    exchange_id = '{exchange_id}'
                ORDER BY time_period_end
                """
            )

            # Ensure that the input data is sorted by time_period_end
            assert token['time_period_end'].is_monotonic_increasing, 'Input data is not sorted by time_period_end'

            features = feature_engineering_pipeline.fit_transform(token)

            # Ensure that the output data is sorted by time_period_end
            assert features['time_period_end'].is_monotonic_increasing, 'Output data is not sorted by time_period_end'

            conn = duckdb.connect(
                database = '/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/database.db',
                read_only = False
            )
            conn.register('features', features)

            try:
                # Upload the file to the database
                QUERY(
                    """
                    INSERT INTO market_data.ml_features 
                    SELECT * FROM features
                    """,
                    conn = conn
                )
            
            # If the table does not exist or the schema has changed, recreate the table
            except Exception as e:
                print(e)
                QUERY(
                    """
                    CREATE OR REPLACE TABLE market_data.ml_features AS
                    SELECT * FROM features
                    """,
                    conn = conn
                )

            conn.unregister('features')
        
    else:
        # X = QUERY(
        #     """
        #     SELECT *
        #     FROM market_data.ml_features
        #     """
        # )
        # X['time_period_end'] = pd.to_datetime(X['time_period_end'])
        # X.sort_values(['asset_id_base', 'asset_id_quote', 'exchange_id', 'time_period_end'], inplace=True)
        X = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet') 
        X['time_period_end'] = pd.to_datetime(X['time_period_end'])
        X.sort_values(['asset_id_base', 'asset_id_quote', 'exchange_id', 'time_period_end'], inplace=True)
        
        return X

    X = QUERY(
        """
        SELECT * FROM market_data.ml_features
        ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
        """
    )
    X['time_period_end'] = pd.to_datetime(X['time_period_end'])

    return X
