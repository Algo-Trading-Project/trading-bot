
# Trading Bot
Repository for Project Poseidon's trading bot.

# Usage

To perform walk-forward optimizations on a set of custom trading strategies, 
run the following command at the root of the package:
<br></br>
<br></br>

```shell
python -m main
```
<br></br>

This runs the main.py file located in the root directory, which performs walk-forward optimization on all the strategies passed into the BackTester object.  The results are uploaded to Redshift for further analysis.
