import pandas as pd
import duckdb
import os
import pathlib

from numba import njit
from utils.db_utils import QUERY
    
def construct_dataset_for_ml(resample_period):
    
    with duckdb.connect(
        database = '~/LocalData/database.db',
        read_only = False
    ) as conn:
        
        # Get all distinct assets in the database
        assets_spot = conn.sql(
        """
        SELECT DISTINCT 
            asset_id_base, 
            asset_id_quote, 
            exchange_id 
        FROM market_data.ohlcv_1m 
        WHERE
            asset_id_quote = 'USDT'
        ORDER BY asset_id_base, asset_id_quote, exchange_id
        """).df()

        assets_futures = conn.sql(
        """
        SELECT DISTINCT 
            asset_id_base, 
            asset_id_quote, 
            exchange_id 
        FROM market_data.futures_ohlcv_1m 
        WHERE
            asset_id_quote = 'USDT'
        ORDER BY asset_id_base, asset_id_quote, exchange_id
        """).df()

        # Create an empty DataFrame to store the final datasets
        dataset_spot = []
        dataset_futures = []

        # Keep track of the number of assets skipped due to missing data
        total_skipped_spot = 0
        total_skipped_futures = 0

        # Loop through each asset
        for i in range(len(assets_spot)):
            print(f"(SPOT) Processing asset {i + 1} of {len(assets_spot)} ({assets_spot.iloc[i]['asset_id_base']}/{assets_spot.iloc[i]['asset_id_quote']} on {assets_spot.iloc[i]['exchange_id']})...")
            # Get the asset
            asset = assets_spot.iloc[i]

            # Get the asset data
            data_spot = conn.sql(
                f"""
                SELECT 
                    time_period_end,
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
                ORDER BY time_period_end
                """
            ).df().set_index('time_period_end').asfreq('1min')
                        
            # Ffill missing values
            categorical_cols = ['asset_id_base', 'asset_id_quote', 'exchange_id']
            for col in categorical_cols:
                mode_spot = data_spot.loc[:,col].mode().iloc[0]
                data_spot.loc[:,col] = data_spot.loc[:,col].fillna(mode_spot)

            # Downsample to resample_period
            data_spot = data_spot.resample(resample_period, label = 'left', closed = 'left').agg({
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
            dataset_spot.append(data_spot)

        for i in range(len(assets_futures)):
            print(f"(FUTURES) Processing asset {i + 1} of {len(assets_futures)} ({assets_futures.iloc[i]['asset_id_base']}/{assets_futures.iloc[i]['asset_id_quote']} on {assets_futures.iloc[i]['exchange_id']})...")
            # Get the asset
            asset = assets_futures.iloc[i]

            # Get the asset data
            data_futures = conn.sql(
                f"""
                SELECT 
                    time_period_end,
                    asset_id_base,
                    asset_id_quote,
                    exchange_id,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    trades
                FROM market_data.futures_ohlcv_1m
                WHERE
                    asset_id_base = '{asset['asset_id_base']}' AND
                    asset_id_quote = '{asset['asset_id_quote']}' AND
                    exchange_id = '{asset['exchange_id']}'
                ORDER BY time_period_end
                """
            ).df().set_index('time_period_end').asfreq('1min')
                        
            # Ffill missing values
            categorical_cols = ['asset_id_base', 'asset_id_quote', 'exchange_id']
            for col in categorical_cols:
                mode_futures = data_futures.loc[:,col].mode().iloc[0]
                data_futures.loc[:,col] = data_futures.loc[:,col].fillna(mode_futures)

            # Downsample to resample_period
            data_futures = data_futures.resample(resample_period, label = 'left', closed = 'left').agg({
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
            dataset_futures.append(data_futures)

        dataset_spot = pd.concat(dataset_spot)
        dataset_futures = pd.concat(dataset_futures)

        dataset_spot = dataset_spot.reset_index()
        dataset_futures = dataset_futures.reset_index()

        dataset_spot['time_period_end'] = pd.to_datetime(dataset_spot['time_period_end'])
        dataset_futures['time_period_end'] = pd.to_datetime(dataset_futures['time_period_end'])

        print(f"Total assets skipped due to missing data: {total_skipped_spot + total_skipped_futures}")
        print(f'Percentage of assets skipped: {(total_skipped_spot + total_skipped_futures) / len(assets_spot + assets_futures) * 100:.2f}%')

        # Save if the spot data file does not exist
        path_spot = f'/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset_{resample_period}.csv'
        dataset_spot.to_csv(path_spot, index = False)
        QUERY(
            f"""
            CREATE OR REPLACE TABLE market_data.ml_dataset_{resample_period} as from read_csv('{path_spot}');
            """
        )

        # Save if the futures data file does not exist
        path_futures = f'/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset_futures_{resample_period}.csv'
        dataset_futures.to_csv(path_futures, index = False)
        QUERY(
            f"""
            CREATE OR REPLACE TABLE market_data.ml_dataset_futures_{resample_period} as from read_csv('{path_futures}');
            """
        )

def get_ml_dataset(resample_period = '1d'):
    # Construct the dataset for machine learning
    construct_dataset_for_ml(resample_period = resample_period)

def get_ml_features(feature_engineering_pipeline):
    tokens = QUERY(
        """
        SELECT DISTINCT asset_id_base, asset_id_quote, exchange_id
        FROM market_data.ml_dataset_1d
        WHERE asset_id_base || '_' || asset_id_quote || '_' || exchange_id IN (        
        SELECT DISTINCT asset_id_base || '_' || asset_id_quote || '_' || exchange_id AS symbol_id
        FROM market_data.spot_trade_features_rolling
        )
        ORDER BY asset_id_base, asset_id_quote, exchange_id
        """
    )

    n = len(tokens[['asset_id_base', 'asset_id_quote', 'exchange_id']])
    prev_data = None
    canonical_cols = None 
    for idx, row in tokens[['asset_id_base', 'asset_id_quote', 'exchange_id']].iterrows():
        asset_id_base = row['asset_id_base']
        asset_id_quote = row['asset_id_quote']
        exchange_id = row['exchange_id']

        symbol_id = f'{asset_id_base}_{asset_id_quote}_{exchange_id}'
        print(f'Processing symbol_id: {symbol_id} ({idx + 1}/{n})')

        token_spot = QUERY(
            f"""
            SELECT 
                asset_id_base || '_' || asset_id_quote || '_' || exchange_id AS symbol_id,
                *
            FROM market_data.ml_dataset_1d
            WHERE
                asset_id_base = '{asset_id_base}' AND
                asset_id_quote = '{asset_id_quote}' AND
                exchange_id = '{exchange_id}'
            ORDER BY time_period_end
            """
        )
        token_futures = QUERY(
            f"""
            SELECT 
                asset_id_base || '_' || asset_id_quote || '_' || exchange_id AS symbol_id,
                *
            FROM market_data.ml_dataset_futures_1d
            WHERE
                asset_id_base = '{asset_id_base}' AND
                asset_id_quote = '{asset_id_quote}' AND
                exchange_id = '{exchange_id}'
            ORDER BY time_period_end
            """
        )
        merged = pd.merge(token_spot, token_futures, on = 'time_period_end', how = 'left', suffixes = ('_spot', '_futures'))

        merged.rename(columns = {
            'asset_id_base_spot': 'asset_id_base',
            'asset_id_quote_spot': 'asset_id_quote',
            'exchange_id_spot': 'exchange_id',
            'symbol_id_spot': 'symbol_id',
            'time_period_end_spot': 'time_period_end',
        }, inplace = True)
        merged.drop(columns = ['asset_id_base_futures', 'asset_id_quote_futures', 'exchange_id_futures', 'symbol_id_futures', 'time_period_end_futures'], inplace = True, errors = 'ignore')

        # Ensure that the input data is sorted by time_period_end
        assert merged['time_period_end'].is_monotonic_increasing, 'Input data is not sorted by time_period_end'

        features = feature_engineering_pipeline.fit_transform(merged)
        if canonical_cols is None:
            canonical_cols = features.columns.tolist()
        else:
            # Ensure that the features have the same columns as the canonical columns
            features = features[canonical_cols]
            assert set(features.columns) == set(canonical_cols), 'Features do not have the same columns as the canonical columns'

        # Ensure that the output data is sorted by time_period_end
        assert features['time_period_end'].is_monotonic_increasing, 'Output data is not sorted by time_period_end'

        if prev_data is None:
            prev_data = features
        else:
            col_set_difference = (set(features.columns) - set(prev_data.columns)) | (set(prev_data.columns) - set(features.columns))
            print(f'Column set difference: {col_set_difference}')
            print()
            if len(col_set_difference) > 0:
                print(f'Column set difference: {col_set_difference}')
                continue
            else:
                prev_data = features

        # Save the features for this token to a file
        file = f'~/LocalData/data/ml_features/{symbol_id}.parquet'
        features.to_parquet(file, index = False, compression = 'snappy')        
