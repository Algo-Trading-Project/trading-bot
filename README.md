# Trading Bot
Repository for Project Poseidon's trading bot.

Trading bot has the following structure:
```
.
├── README.md
├── dags
│   ├── fetch_eth_block_rewards.py
│   ├── fetch_eth_transaction_gas_used.py
│   ├── fetch_new_eth_data.py
│   ├── fetch_order_book_data_1h.py
│   ├── fetch_quote_data_1h.py
│   ├── fetch_tick_data.py
│   ├── fetch_token_prices_1h.py
│   ├── stg_block_reward_to_prod.py
│   ├── stg_block_to_prod.py
│   ├── stg_gas_used_to_prod.py
│   ├── stg_transaction_to_prod.py
│   └── stg_transfer_to_prod.py
├── plugins
│   ├── __init__.py
│   ├── operators
│   │   ├── __init__.py
│   │   ├── get_block_rewards.py
│   │   ├── get_coinapi_prices_operator.py
│   │   ├── get_eth_transaction_gas_used.py
│   │   ├── get_order_book_data_operator.py
│   │   ├── get_quote_data_operator.py
│   │   ├── get_tick_data_operator.py
│   │   ├── redshift_sql_operator.py
│   │   └── web3_alchemy_to_s3_operator.py
│   └── plugins.zip
├── plugins.zip
└── requirements
    └── requirements.txt

5 directories, 26 files```
