# Trading Bot
Repository for Project Poseidon's trading bot.

Trading bot has the following structure:
.
├── README.md
├── backtest
│   ├── backtester.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── pairs_trading
│   │   │   ├── __init__.py
│   │   │   ├── pairs_trading_backtest.py
│   │   │   └── pairs_trading_numba.py
│   │   ├── performance
│   │   │   ├── __init__.py
│   │   │   └── performance_metrics.py
│   │   ├── simulation
│   │   │   ├── __init__.py
│   │   │   └── monte_carlo.py
│   │   └── wfo
│   │       ├── __init__.py
│   │       └── walk_forward_optimization.py
│   ├── pairs_trading_backtester.py
│   └── strategies
│       ├── README.md
│       ├── __init__.py
│       ├── base_strategy.py
│       ├── bollinger_bands.py
│       └── ma_crossover.py
├── notebooks
│   ├── eda.ipynb
│   ├── simple_ml.ipynb
│   └── simulated_curves.csv
├── simulated_curves.csv
└── trading_bot
    └── __init__.py
    
10 directories, 23 files