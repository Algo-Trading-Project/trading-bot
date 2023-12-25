# order_book_data_processor.py
import pandas as pd
import numpy as np
import redshift_connector

from order_book_snapshot import OrderBookSnapshot

class OrderBookDataProcessor:
    def __init__(self):
        pass

    def get_snapshots(self):
        #TODO: Implement this method to read order book snapshots from Redshift

        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'token_price',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to fetch OHLCV data for a token & exchange of interest
                title = base + '-' + quote
                query = """
                SELECT 
                    time_period_end,
                    price_open,
                    price_high,
                    price_low,
                    price_close,
                    - market_data
                        - order_book_snapshot_processor.py
                        - order_book_data_processor.py
                        - order_book_snapshot.py
                FROM token_price.coinapi.order_book_data_1h
                WHERE 
                    asset_id_base = '{}' AND
                    asset_id_quote = '{}' AND
                    exchange_id = '{}'
                ORDER BY time_period_start ASC
                """.format(base, quote, exchange)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()

        test_snapshot = '{"symbol_id": "COINBASE_SPOT_BTC_USD", "time_exchange": "2021-01-01T00:00:00.0000000Z", "time_coinapi": "2021-01-01T00:00:01.0000000Z", "bids": "{\\"price\\": 29000.5, \\"size\\": 0.5} {\\"price\\": 29000.4, \\"size\\": 0.7}", "asks": "{\\"price\\": 29001.5, \\"size\\": 0.5} {\\"price\\": 29001.6, \\"size\\": 0.3}"}'
        return [test_snapshot]

    def process_snapshots(self):
        """
        Process the snapshots in the file and return a pandas DataFrame.
        """

        snapshots = self.get_snapshots()
        
        data = []
        timestamps = []

        previous_order_book = None

        for snapshot_str in snapshots:
            if not snapshot_str.strip():
                continue  # Skip empty lines

            snapshot = OrderBookSnapshot(snapshot_str)
            order_book_metrics = {
                'average_bid_price': snapshot.average_price('bid'),
                'average_ask_price': snapshot.average_price('ask'),
                'total_bid_volume': snapshot.total_volume('bid'),
                'total_ask_volume': snapshot.total_volume('ask'),
                'spread': snapshot.spread(),
                'bid_weighted_avg_price': snapshot.weighted_average_price('bid'),
                'ask_weighted_avg_price': snapshot.weighted_average_price('ask'),
                'order_book_imbalance': snapshot.order_book_imbalance(),
                'volatility_estimate': snapshot.volatility_estimate(),
                'asymmetry_index_10': snapshot.asymmetry_index(10),
                'concentration_score_5_percent': snapshot.concentration_score(0.05),
                'bid_slippage_500_units': snapshot.price_slippage(500, 'sell'),
                'ask_slippage_500_units': snapshot.price_slippage(500, 'buy'),
                'cumulative_bid_depth_10': snapshot.cumulative_depth(10, 'bid'),
                'cumulative_ask_depth_10': snapshot.cumulative_depth(10, 'ask'),
                'liquidity_fluctuation_index': np.nan if previous_order_book is None else snapshot.liquidity_fluctuation_index(previous_order_book)
            }
            
            data.append(order_book_metrics)
            timestamps.append(snapshot.timestamp)

            previous_order_book = snapshot

        df = pd.DataFrame(data, index = pd.to_datetime(timestamps))

        return df

# Optionally include test functions or other auxiliary methods as needed.
def test_order_book_processing():

    # Create an instance of the processor and process the test string
    processor = OrderBookDataProcessor()
    df = processor.process_snapshots()

    # Check if the DataFrame is not empty and has the expected columns
    assert not df.empty, "DataFrame is empty"

    expected_columns = ['average_bid_price', 'average_ask_price', 'total_bid_volume', 'total_ask_volume', 'spread']
    
    for column in expected_columns:
        assert column in df.columns, f"Missing expected column: {column}"

    print("Test passed successfully!")


if __name__ == '__main__':
    test_order_book_processing()