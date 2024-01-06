import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import vectorbt as vbt
import redshift_connector



def calculate_triple_barrier_labels(ohlcv_df, atr_window, max_holding_time):
    # Calculate ATR using vectorbt
    high = ohlcv_df['price_high']
    low = ohlcv_df['price_low']
    close = ohlcv_df['price_close']

    atr = vbt.ATR.run(high, low, close, window=atr_window).atr

    # Initialize labels array with zeros
    labels = np.zeros(len(close))

    # Loop through the close price series
    for i in range(len(close) - max_holding_time):
        # Set the upper and lower barriers
        upper_barrier = close[i] + atr[i]
        lower_barrier = close[i] - atr[i]
        barrier_hit = False

        # Check if price hits the barriers within the max holding time
        for j in range(i + 1, i + max_holding_time + 1):
            # Ensure the loop does not go beyond the length of the series
            if j >= len(close):
                break

            if high[j] >= upper_barrier:
                labels[i] = 1  # Upper barrier hit
                barrier_hit = True
                break
            elif low[j] <= lower_barrier:
                labels[i] = -1  # Lower barrier hit
                barrier_hit = True
                break

        # If no barriers are hit, determine label based on closing price after max holding time
        if not barrier_hit:
            labels[i] = 1 if close[j] > close[i] else -1

    # Create a Pandas Series from the labels array
    label_series = pd.Series(labels, index=ohlcv_df.index)

    return label_series

def execute_query(query, cols, date_col):
    with redshift_connector.connect(
        host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
        database = 'token_price',
        user = 'administrator',
        password = 'Free2play2'
    ) as conn:
        with conn.cursor() as cursor:
            # Execute query on Redshift and return result
            cursor.execute(query)
            tuples = cursor.fetchall()
            
            # Return queried data as a DataFrame
            df = pd.DataFrame(tuples, columns = cols).set_index(date_col)

            # For each column in DataFrame, convert to float, skip if not possible
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except:
                    continue

            # Fill in any gaps in data with last seen value
            df = df.drop_duplicates().asfreq(freq = 'H', method = 'ffill')

            return df

def get_ohlcv_data(base, quote, exchange):
    query = """
    SELECT
        time_period_end,
        price_open,
        price_high,
        price_low,
        price_close,
        volume_traded
    FROM token_price.coinapi.price_data_1h
    WHERE 
        asset_id_base = '{}' AND
        asset_id_quote = '{}' AND
        exchange_id = '{}'
    ORDER BY time_period_start ASC
    """.format(base, quote, exchange)

    ohlcv_1h_cols = ['time_period_end', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']
    ohlcv_1h = execute_query(
        query = query,
        cols = ohlcv_1h_cols,
        date_col = 'time_period_end'
    )

    return ohlcv_1h

def calculate_train_performance(X_train, y_train, model):
    # Train Predictions
    y_pred = model.predict(X_train)

    # Classification report
    print('Train Classification Report')
    print()
    print(classification_report(y_train, y_pred))

    # Various metrics rounded to 4 decimal places
    accuracy = round(accuracy_score(y_train, y_pred), 4)
    precision = round(precision_score(y_train, y_pred, average = 'weighted'), 4)
    recall = round(recall_score(y_train, y_pred, average = 'weighted'), 4)
    f1 = round(f1_score(y_train, y_pred, average = 'weighted'), 4)
    roc_auc = round(roc_auc_score(y_train, y_pred, average = 'weighted'), 4)

    # Print various metrics
    print('Train Accuracy: ', accuracy)
    print('Train Precision: ', precision)
    print('Train Recall: ', recall)
    print('Train F1 Score: ', f1)
    print('Train ROC AUC Score: ', roc_auc)

    # Plot horizontal bar plot of top 20 most important features
    feature_importances = pd.Series(model.feature_importances_, index = X_train.columns)
    feature_importances.nlargest(20).sort_values().plot(
        kind = 'barh', 
        figsize = (10, 6), 
        title = 'Top 20 Most Important Features',
        color = 'green',
        xlabel = 'Feature Importance',
        fontsize = 14,
        width = 0.8,
        edgecolor = 'black',
        linewidth = 1.2
    )

def calculate_test_performance(X_test, y_test, model):
    # Test predictions
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:,1]

    # Classification report
    print('Test Classification Report')
    print()
    print(classification_report(y_test, y_pred))

    # Various metrics rounded to 4 decimal places
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, average = 'weighted'), 4)
    recall = round(recall_score(y_test, y_pred, average = 'weighted'), 4)
    f1 = round(f1_score(y_test, y_pred, average = 'weighted'), 4)
    roc_auc = round(roc_auc_score(y_test, y_pred, average = 'weighted'), 4)

    # Print various metrics
    print('Test Accuracy: ', accuracy)
    print('Test Precision: ', precision)
    print('Test Recall: ', recall)
    print('Test F1 Score: ', f1)
    print('Test ROC AUC Score: ', roc_auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    plt.figure(figsize = (10, 4))
    plt.plot(fpr, tpr, color = 'green', label = 'ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'red', linestyle = '--', label = 'Random Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
