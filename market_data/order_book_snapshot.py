import re

class OrderBookSnapshot:
    
    def __init__(self, snapshot_str):
        self.timestamp, self.bids, self.asks = self.parse_snapshot(snapshot_str)
        
        self.bids = sorted(self.bids, key=lambda x: x['price'], reverse=True)[:20]  # Top 20 bids, sorted in descending order
        self.asks = sorted(self.asks, key=lambda x: x['price'])[:20]  # Top 20 asks, sorted in ascending order

    @staticmethod
    def parse_timestamp(snapshot_str):
        
        """
        Extracts the timestamp from the snapshot string.

        :param snapshot_str: A string containing the order book snapshot.
        :return: Timestamp as a string.
        """
        timestamp_pattern = r'"time_exchange": "([^"]+)"'
        match = re.search(timestamp_pattern, snapshot_str)
        return match.group(1) if match else None

    @staticmethod
    def parse_orders(orders_str):
        """
        Parses a string of orders into a list of dictionaries.

        :param orders_str: A string containing orders.
        :return: A list of order dictionaries.
        """
        order_pattern = r'\{\\"price\\": (\d+\.\d+), \\"size\\": (\d+\.\d+)\}'
        orders = re.findall(order_pattern, orders_str)
        return [{"price": float(price), "size": float(size)} for price, size in orders]

    def parse_snapshot(self, snapshot_str):
        """
        Parses a single order book snapshot string into timestamp, bids, and asks.

        :param snapshot_str: A string containing the order book snapshot.
        :return: Timestamp, and two lists containing parsed bids and asks.
        """
        timestamp = self.parse_timestamp(snapshot_str)
        bids_str = snapshot_str.split('"bids": ')[1].split(', "asks": ')[0]
        asks_str = snapshot_str.split('"asks": ')[1]

        bids = self.parse_orders(bids_str)
        asks = self.parse_orders(asks_str)

        return timestamp, bids, asks

    def average_price(self, order_type):
        orders = self.bids if order_type == 'bid' else self.asks
        total_price = sum(order['price'] * order['size'] for order in orders)
        total_quantity = sum(order['size'] for order in orders)
        return total_price / total_quantity if total_quantity else 0

    def total_volume(self, order_type):
        orders = self.bids if order_type == 'bid' else self.asks
        return sum(order['size'] for order in orders)

    def spread(self):
        best_bid = self.bids[0]['price'] if self.bids else 0
        best_ask = self.asks[0]['price'] if self.asks else 0
        return best_ask - best_bid

    def weighted_average_price(self, order_type):
        orders = self.bids if order_type == 'bid' else self.asks
        total_volume = self.total_volume(order_type)
        if total_volume == 0:
            return 0
        return sum(order['price'] * order['size'] for order in orders) / total_volume

    def order_book_imbalance(self):
        total_bid_volume = self.total_volume('bid')
        total_ask_volume = self.total_volume('ask')
        total_volume = total_bid_volume + total_ask_volume
        return total_bid_volume / total_volume if total_volume > 0 else 0

    def price_slippage(self, volume, order_type):
        orders = self.asks if order_type == 'buy' else self.bids
        remaining_volume = volume
        weighted_price = 0
        for order in orders:
            trade_quantity = min(order['size'], remaining_volume)
            weighted_price += trade_quantity * order['price']
            remaining_volume -= trade_quantity
            if remaining_volume <= 0:
                break
        return weighted_price / volume if volume > 0 else 0

    def cumulative_depth(self, depth, order_type):
        orders = self.bids if order_type == 'bid' else self.asks
        return sum(order['size'] for order in orders[:depth])

    def asymmetry_index(self, depth):
        bid_volume = self.cumulative_depth(depth, 'bid')
        ask_volume = self.cumulative_depth(depth, 'ask')
        return (bid_volume - ask_volume) / (bid_volume + ask_volume) if bid_volume + ask_volume > 0 else 0

    def concentration_score(self, percentage):
        best_bid = self.bids[0]['price'] if self.bids else 0
        best_ask = self.asks[0]['price'] if self.asks else 0
        bid_threshold = best_bid * (1 - percentage)
        ask_threshold = best_ask * (1 + percentage)
        bid_concentration = sum(order['size'] for order in self.bids if order['price'] >= bid_threshold)
        ask_concentration = sum(order['size'] for order in self.asks if order['price'] <= ask_threshold)
        total_concentration = bid_concentration + ask_concentration
        total_volume = self.total_volume('bid') + self.total_volume('ask')
        return total_concentration / total_volume if total_volume > 0 else 0

    def volatility_estimate(self):
        spread = self.spread()
        depth_bid = self.cumulative_depth(5, 'bid')
        depth_ask = self.cumulative_depth(5, 'ask')
        depth_metric = (depth_bid + depth_ask) / 2
        return spread / depth_metric if depth_metric > 0 else 0

    def liquidity_fluctuation_index(self, previous_order_book):
        current_depth = self.cumulative_depth(10, 'bid') + self.cumulative_depth(10, 'ask')
        previous_depth = previous_order_book.cumulative_depth(10, 'bid') + previous_order_book.cumulative_depth(10, 'ask')
        return (current_depth - previous_depth) / previous_depth if previous_depth > 0 else 0

