from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Numba implementation of the permutation test
@njit(parallel = True)
def permutation_test(y_pred, trade_returns, n_simulations = 1000):
    results = np.zeros(n_simulations)

    for i in prange(n_simulations):
        # Shuffle the predictions (not in place)
        y_pred_permuted = y_pred.copy()
        np.random.shuffle(y_pred_permuted)

        # Calculate the sample statistic
        sample_means = np.zeros(2)
        sample_means[0] = np.mean(trade_returns[y_pred_permuted == 0])
        sample_means[1] = np.mean(trade_returns[y_pred_permuted == 1])

        results[i] = sample_means[1] - sample_means[0]

    return results

# Numba implementation of the bootstrap distribution
@njit(parallel = True)
def calculate_bootstrap_distribution_numba(X_test, n_simulations = 1000):
    diff_distrubtion = np.zeros(n_simulations)
    pos_pred_mean_returns = np.zeros(n_simulations)
    neg_pred_mean_returns = np.zeros(n_simulations)

    for i in prange(n_simulations):
        # Sample with replacement
        X_test_sample = X_test[np.random.randint(0, X_test.shape[0], size = X_test.shape[0]), :]

        # Calculate the mean returns for positive and negative predictions
        mean_returns_pos = np.mean(X_test_sample[X_test_sample[:, 0] == 1, 1])
        mean_returns_neg = np.mean(X_test_sample[X_test_sample[:, 0] == 0, 1])

        pos_pred_mean_returns[i] = mean_returns_pos
        neg_pred_mean_returns[i] = mean_returns_neg

        # Calculate the sample statistic
        diff_distrubtion[i] = mean_returns_pos - mean_returns_neg

    return diff_distrubtion, pos_pred_mean_returns, neg_pred_mean_returns

# Function to run the OOS statistical tests
def run_oos_statistical_tests(y_pred, trade_returns, n_simulations, sample_statistic):
    # Calculate the permutation test
    permutation_results = permutation_test(y_pred, trade_returns, n_simulations = n_simulations)
    empirical_p_value = (permutation_results >= sample_statistic).mean()

    # Calculate the bootstrap distribution
    mean_returns_diff_dist, mean_returns_pos_pred_dist, mean_returns_neg_pred_dist = calculate_bootstrap_distribution_numba(np.column_stack((y_pred, trade_returns)), n_simulations = n_simulations)

    # Calculate the confidence interval for each distribution
    ci_mean_returns_diff = np.percentile(mean_returns_diff_dist, [0.5, 99.5])
    ci_mean_returns_diff[0] = round(ci_mean_returns_diff[0], 3)
    ci_mean_returns_diff[1] = round(ci_mean_returns_diff[1], 3)
    median_mean_returns_diff = np.median(mean_returns_diff_dist)

    ci_mean_pos_pred = np.percentile(mean_returns_pos_pred_dist, [0.5, 99.5])
    ci_mean_pos_pred[0] = round(ci_mean_pos_pred[0], 3)
    ci_mean_pos_pred[1] = round(ci_mean_pos_pred[1], 3)
    median_mean_pos_pred = np.median(mean_returns_pos_pred_dist)

    ci_mean_neg_pred = np.percentile(mean_returns_neg_pred_dist, [0.5, 99.5])
    ci_mean_neg_pred[0] = round(ci_mean_neg_pred[0], 3)
    ci_mean_neg_pred[1] = round(ci_mean_neg_pred[1], 3)
    median_mean_neg_pred = np.median(mean_returns_neg_pred_dist)

    fig, ax = plt.subplots(2, 2, figsize = (18, 6))

    # Plot the permutation test distribution
    sns.histplot(permutation_results, kde = True, color = 'black', alpha = 0.7, ax = ax[0][0])
    ax[0][0].axvline(x = sample_statistic, color = 'red', linestyle = '--', label = 'OOS test statistic')
    ax[0][0].set_title(f'Permutation Test Distribution of OOS Mean Return Differences (N = {n_simulations:,}), p = {empirical_p_value:.4f}')
    ax[0][0].set_xlabel('Mean Trade Returns Difference')
    ax[0][0].set_ylabel('Frequency')
    ax[0][0].legend()

    # Plot the bootstrap distribution
    sns.histplot(mean_returns_diff_dist, kde = True, color = 'blue', alpha = 0.7, ax = ax[0][1])
    ax[0][1].axvline(x = ci_mean_returns_diff[0], color = 'red', linestyle = '--', label = '0.5th Percentile')
    ax[0][1].axvline(x = median_mean_returns_diff, color = 'black', linestyle = '--', label = 'Median')
    ax[0][1].axvline(x = ci_mean_returns_diff[1], color = 'green', linestyle = '--', label = '99.5th Percentile')
    ax[0][1].set_title(f'Bootstraped Distribution of OOS Mean Return Differences (99% CI {ci_mean_returns_diff}, N = {n_simulations:,})')
    ax[0][1].set_xlabel('Mean Trade Returns Difference')
    ax[0][1].set_ylabel('Frequency')
    ax[0][1].legend()

    # Plot the distribution of mean returns for positive predictions
    sns.histplot(mean_returns_pos_pred_dist, kde = True, color = 'green', alpha = 0.7, ax = ax[1][0])
    ax[1][0].axvline(x = ci_mean_pos_pred[0], color = 'red', linestyle = '--', label = '0.5th Percentile')
    ax[1][0].axvline(x = median_mean_pos_pred, color = 'blue', linestyle = '--', label = 'Median')
    ax[1][0].axvline(x = ci_mean_pos_pred[1], color = 'green', linestyle = '--', label = '99.5th Percentile')
    ax[1][0].set_title(f'Bootstrapped Distribution of Mean Returns for Positive Predictions (99% CI {ci_mean_pos_pred}, N = {n_simulations:,})')
    ax[1][0].set_xlabel('Mean Trade Returns')
    ax[1][0].set_ylabel('Frequency')
    ax[1][0].legend()

    # Plot the distribution of mean returns for negative predictions
    sns.histplot(mean_returns_neg_pred_dist, kde = True, color = 'red', alpha = 0.7, ax = ax[1][1])
    ax[1][1].axvline(x = ci_mean_neg_pred[0], color = 'red', linestyle = '--', label = '0.5th Percentile')
    ax[1][1].axvline(x = median_mean_neg_pred, color = 'blue', linestyle = '--', label = 'Median')
    ax[1][1].axvline(x = ci_mean_neg_pred[1], color = 'green', linestyle = '--', label = '99.5th Percentile')
    ax[1][1].set_title(f'Bootstrapped Distribution of Mean Returns for Negative Predictions (99% CI {ci_mean_neg_pred}, N = {n_simulations:,})')
    ax[1][1].set_xlabel('Mean Trade Returns')
    ax[1][1].set_ylabel('Frequency')
    ax[1][1].legend()

    plt.tight_layout()
    plt.legend()
    plt.show()