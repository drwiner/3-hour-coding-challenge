import pandas as pd
import math
from typing import List

def plogp(df: pd.DataFrame, target_col: str, target_value: str) -> float:
    """ Calculate plogp for dataframe for observations matching target """
    # Get observations matching target
    df_matching_target = df[df[target_col] == target_value]

    if df_matching_target.shape[0] == 0:
        return 0

    if df.shape[0] == 0:
        return 0

    prob: float = df_matching_target.shape[0] / df.shape[0]

    if prob == 0:
        return 0

    # This is currently handling edge case where math.log(1, 1) is undefined
    if prob == 1 and df.shape[0] == 1:
        return 1

    return prob * math.log(prob, len(df))


def entropy(df: pd.DataFrame, target_col: str) -> float:
    """ Calculate entropy for dataframe """
    # Get unique values for target column
    unique_values = df[target_col].unique()

    # Calculate plogp for each unique value
    return - sum(plogp(df, target_col, value) for value in unique_values)


def information_gain(obs_df: pd.DataFrame, target_col: str, feature_col: str) -> float:
    """ Calculate information gain for dataframe for specific feature col"""

    gain: float = 0.0

    unique_values = set(obs_df[feature_col].unique())

    for unique_value in unique_values:
        split_df = obs_df[obs_df[feature_col] == unique_value]
        if obs_df.shape[0] == 0:
            continue
        gain += (split_df.shape[0] / obs_df.shape[0]) * entropy(split_df, target_col)

    # Gain is total entropy minus gain over each possible value
    return entropy(obs_df, target_col) - gain


def pick_best_feature(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> str:
    """ Pick the best feature to split on """
    best_feature = None
    best_gain = -math.inf

    for feature in feature_cols:
        gain = information_gain(df, target_col, feature)

        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature


if __name__ == "__main__":
    # Test plogp
    df = pd.DataFrame(
        {
            "Name": ["cat", "cat", "cat", "dog", "dog", "dog"],
            "Number of legs": [3, 3, 4, 4, 5, 4],
            "Color": ["white", "black", "black", "black", "black", "white"],
        }
    )

    print(plogp(df, "Name", "cat"))


    # Test entropy
    print(entropy(df, "Name"))

    # Test information gain
    print(information_gain(df, "Name", "Color"))

    # Trying to test another edge case but won't worrya bout it for now.
    df2 = pd.DataFrame(
        {
            "Name": ["cat", "cat", "cat", "dog", "dog", "dog"],
            "Number of legs": [4, 4, 4, 4, 4, 4],
            "Color": ["white", "black", "black", "black", "green", "blue"],
        }
    )
    print(information_gain(df2, "Name", "Number of legs"))

    # Test pick best feature
    print(pick_best_feature(df, "Name", ["Number of legs", "Color"]))
    print(pick_best_feature(df2, "Name", ["Number of legs", "Color"]))