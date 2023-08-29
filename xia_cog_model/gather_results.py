import os
import glob
import pandas as pd
import numpy as np
import concurrent.futures
from functools import partial
from typing import Callable


# Function to load a single CSV file and compute summary statistics
def load_csv(
    file_path: str, participant: int, summary_func: Callable
) -> pd.DataFrame:
    # Load the data, skipping the first 45 lines
    df = pd.read_csv(file_path, skiprows=45)

    # Compute the summary statistics for r, v, k
    r_columns = [col for col in df.columns if col.startswith("r.")]
    v_columns = [col for col in df.columns if col.startswith("v.")]
    summary_r = df[r_columns].apply(summary_func)
    summary_v = df[v_columns].apply(summary_func)
    summary_k = df["k"].dropna().apply(summary_func).mean()

    # Return a DataFrame with the results
    result = pd.DataFrame(
        {
            "participant": participant,
            "trial": np.arange(1, len(summary_r) + 1),
            "r": summary_r.values,
            "v": summary_v.values,
            "k": [summary_k] * len(summary_r),
        }
    )
    return result


# Function to load all CSV files in a directory and compute summary statistics
def load_all_csvs(
    directory: str, participant: int, summary_func: Callable = np.mean
) -> pd.DataFrame:
    # Get a list of all CSV files in the directory
    files = glob.glob(f"{directory}/*.csv")

    # Initialize an empty list to store the results
    results = []

    # Use a thread pool to process the files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        func = partial(
            load_csv, participant=participant, summary_func=summary_func
        )

        # Process the files in parallel
        for result in executor.map(func, files):
            # Append the result to the list
            results.append(result)

    # Concatenate all results into a single DataFrame
    return pd.concat(results)


# Function to process all participants in a directory
def process_all_participants(
    base_path: str, summary_func: Callable = np.mean
) -> pd.DataFrame:
    # Get a list of all subdirectories in the base path
    subdirectories = [f.path for f in os.scandir(base_path) if f.is_dir()]

    # Initialize an empty list to store the results
    results = []

    # Process each subdirectory in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Prepare the arguments for the load_all_csvs function
        args = [
            (subdir, int(os.path.basename(subdir).split("_")[-1]), summary_func)
            for subdir in subdirectories
        ]

        # Process the subdirectories in parallel
        for result in executor.map(lambda x: load_all_csvs(*x), args):
            # Append the result to the list
            results.append(result)

    # Concatenate all results into a single DataFrame
    result_df = pd.concat(results)

    # Save the result DataFrame to CSV and Excel files
    result_df.to_csv(f"{base_path}/all_participants.csv", index=False)
    result_df.to_excel(f"{base_path}/all_participants.xlsx", index=False)

    return result_df
