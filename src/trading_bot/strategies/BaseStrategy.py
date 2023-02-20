import pandas as pd
import redshift_connector
import requests as r

from src.trading_bot.Trade import Trade
# import warnings
# warnings.filterwarnings("ignore")

class BaseStrategy:

    def __init__(self, strat_time_frame_minutes, sl, symbol_id):
        self.symbol_id = symbol_id
        self.strat_time_frame_minutes = strat_time_frame_minutes
        self.sl = sl
        # self.data = self.__fetch_historical_price_data()
        self.data = pd.DataFrame([], columns = ['time_period_start', 'time_period_end', 'price_open',
                                                     'price_high', 'price_low', 'price_close',
                                                     'volume_traded'])
        self.position = None
        self.curr_price = None
        self.tick_count = 0

    def __fetch_historical_price_data(self):
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'administrator',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to fetch open price data for an ETH pair & exchange of interest
                # within a specified date range
                exchange, _, base, quote = self.symbol_id.split('_')
                query = """
                SELECT
                    time_period_start,
                    time_period_end,
                    price_open,
                    price_high,
                    price_low,
                    price_close,
                    volume_traded
                FROM token_price.eth.stg_price_data_1h
                WHERE
                    asset_id_base = '{}' AND
                    asset_id_quote = '{}' AND
                    exchange_id = '{}'
                ORDER BY time_period_start DESC
                LIMIT 168
                """.format(base, quote, exchange)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()

                df = pd.DataFrame(tuples, columns = ['time_period_start', 'time_period_end', 'price_open',
                                                     'price_high', 'price_low', 'price_close',
                                                     'volume_traded'])
                df = df.sort_values(by = ['time_period_start'])
                
                latest_ds_for_pair_redshift = df.iloc[-1]['time_period_end'].isoformat()
                api_request_url = 'https://rest.coinapi.io/v1/ohlcv/{}/history?period_id={}&time_start={}&limit={}'.format(self.symbol_id, '1HRS', latest_ds_for_pair_redshift, 100000)
                headers = {'X-CoinAPI-Key':'34FA476E-CCCB-4AC7-A033-B472113BBD22'}
        
                response = r.get(
                    url = api_request_url,
                    headers = headers,
                )
                response_json = response.json()

                historical_data = pd.DataFrame(response_json, columns = ['time_period_start', 'time_period_end', 'time_open',
                                                                         'time_close', 'price_open', 'price_high',
                                                                         'price_low', 'price_close', 'volume_traded', 'trades_count',
                                                                         ])
                historical_data = historical_data.drop(['time_open', 'time_close', 'trades_count'], axis = 1)
                df = df.append(historical_data, ignore_index = True)
                
                return df
            
    def __generate_trading_signal(self):
        if self.position and self.curr_price <= self.position.price - (self.position.price * self.sl):
            return 'Sell'
        else:
            return 'None'
        
    def add_position(self, tick_data):
        if self.position != None:
            return False
        
        exchange, _, base, quote = self.symbol_id.split('_')
        self.position = Trade(base, quote, exchange, tick_data['price_open'], 1, tick_data['time_period_start'], 'BUY')

        return True

    def remove_position(self):
        if self.position == None:
            return False
        
        self.position = None

        return True

    def process_tick(self, tick_data):
        self.curr_price = float(tick_data['price_open'])

        strat_time_frame_flag = False

        if self.tick_count % self.strat_time_frame_minutes == 0:
            self.data = self.data.append(tick_data, ignore_index = True)
            strat_time_frame_flag = True

        trading_signal = self.__generate_trading_signal()

        self.tick_count += 1

        if trading_signal == 'Sell':
            return trading_signal
        
        elif trading_signal == 'None' and strat_time_frame_flag:
            return 'Generate'

        elif trading_signal == 'None' and not strat_time_frame_flag:
            return 'None'

        
