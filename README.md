# Trading Bot
Repository for Project Poseidon's trading bot.

Trading bot has the following structure:
```
.
├── README.md
|
├── backtest
|   |   
│   ├── backtester.py
|   |   
│   ├── core
|   |   |  
│   │   ├── __init__.py
|   |   |  
│   │   ├── pairs_trading
|   |   |   |
│   │   │   ├── __init__.py
│   │   │   ├── pairs_trading_backtest.py
│   │   │   └── pairs_trading_numba.py
|   |   |  
│   │   ├── performance
│   │   │   ├── __init__.py
│   │   │   ├── pbo.py
│   │   │   └── performance_metrics.py
|   |   |  
│   │   ├── simulation
│   │   │   ├── __init__.py
│   │   │   └── monte_carlo.py
|   |   |
│   │   └── wfo
│   │       ├── __init__.py
│   │       └── walk_forward_optimization.py
|   |
│   ├── pairs_trading_backtester.py
|   |   
│   └── strategies
│       ├── README.md
│       ├── __init__.py
│       ├── base_strategy.py
│       ├── bollinger_bands.py
│       ├── linear_regression.py
│       └── ma_crossover.py
|
├── market_data
│   ├── __init__.py
│   ├── order_book_snapshot.py
│   └── order_book_snapshot_processor.py
|
├── notebooks
│   └── eda.ipynb
|
└── trading_bot
    └── __init__.py

11 directories, 25 files, 4188 lines of code
```
