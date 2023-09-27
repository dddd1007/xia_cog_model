import os
import glob
import pandas as pd
from scipy.stats import entropy
import numpy as np


def load_dataframes(tmp_path, skiprows=45):
    """
    Load dataframes from csv files in a directory.

    :param tmp_path: The directory path where the csv files are located.
    :param skiprows: The number of rows to skip at the beginning of each file. Default is 45.
    :return: A concatenated dataframe of all csv files.
    """
    csv_files = glob.glob(os.path.join(tmp_path, "*.csv"))
    dataframes = [pd.read_csv(file, skiprows=skiprows, dtype={0: str}) for file in csv_files]
    df = pd.concat(dataframes, ignore_index=True)
    df = df.dropna()
    return df


def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    :param p: The first probability distribution.
    :param q: The second probability distribution.
    :return: The Kullback-Leibler divergence between p and q.
    """
    return entropy(p, q)


def param_kl(df, param, debug=False):
    """
    Calculate the Kullback-Leibler divergence for a specific parameter across all columns in a dataframe.

    :param df: The dataframe containing the data.
    :param param: The parameter to calculate the Kullback-Leibler divergence for.
    :param debug: If True, print debug information. Default is False.
    :return: A dictionary containing the original, center, and z-score normalized Kullback-Leibler divergence results.
    """
    param_cols = [col for col in df.columns if col.startswith(param)]
    param_count = len(param_cols)

    if debug:
        print(f"param count: {param_count}")
        print(f"First five param cols: {param_cols[:5]}")

    kl_result = np.zeros(param_count)
    for i in range(param_count):
        kl_result[i - 1] = kl_divergence(df[param_cols[i - 1]], df[param_cols[i]])

        if debug:
            print(
                f"KL divergence for {param_cols[i-1]} and {param_cols[i]}: {kl_result[i]}"
            )

    # normalize kl_result

    # 0-1 normalization
    kl_result_01 = (kl_result - np.min(kl_result)) / (
        np.max(kl_result) - np.min(kl_result)
    )

    # z-score normalization
    kl_result_zcore = (kl_result - np.mean(kl_result)) / np.std(kl_result)

    result_dict = {
        "original_kl": kl_result,
        "center_kl": kl_result_01,
        "zscore_kl": kl_result_zcore,
    }

    return result_dict
