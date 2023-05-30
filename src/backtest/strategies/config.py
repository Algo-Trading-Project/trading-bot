from strategies.TestStratVBT import TestStratVBT

# This file will be used as a place to configure all of the backtest
# inputs that will be used in BackTester.py

# This section contains one dictionary.  The keys are the custom indicator functions
# or strategies that will be backtested and the values are dictionaries continaing
# all hyperparameter combinations to be backtested on

optimize_dict = {
    TestStratVBT.indicator_func: TestStratVBT.optimize_dict
}

# This section contains one dictionary.  The keys are the custom indicator functions
# or strategies that will be backtested and the values are dictionaries with the following
# keys:
#       class_name : str - Name of the strategy
#       short_name : str - Shorter name of the strategy
#       input_names : list - List of inputs to the strategy
#       param_names : list - List of hyperparameters of the strategy
#       output_names : list - List of outputs of the strategy

indicator_factory_dict = {
    TestStratVBT.indicator_func: TestStratVBT.indicator_factory_dict
}

# This section contains one dictionary.  The keys are the custom indicator functions
# or strategies that will be backtested and the values are dictionaries containing
# default values for every hyperparameter of a given strategy

indicator_func_defaults_dict = {
    TestStratVBT.indicator_func: TestStratVBT.default_dict
}
