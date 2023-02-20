

class Trade:
    def __init__(self, base, quote, exchange, price, amount, trade_date, trade_type):
        self.base = base
        self.quote = quote
        self.exchange = exchange
        self.price = price
        self.amount = amount
        self.trade_date = trade_date
        self.trade_type = trade_type
