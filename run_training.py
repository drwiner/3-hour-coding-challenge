import argparse
import os
import pandas as pd
import logging
from src.id3 import id3_algo
import json


# from observations import FeatureType
#
#
# REQUIRED_COLS = {
#     'Name': FeatureType.ANIMAL,
#     'Number of legs': FeatureType.NUM_LEGS,
#     'Color': FeatureType.COLOR
# }


def do_check_on_input(input_df: pd.DataFrame) -> None:
    """Do some checks on the input dataframe

    Args:
        input_df (pd.DataFrame): Input whose columns we have to check.

    Raises:
        ValueError: If the input dataframe does not have the required columns or is empty.
    """
    if input_df.empty:
        raise ValueError('Input dataframe is empty.')

    # if not all(col in input_df.columns for col in REQUIRED_COLS.keys()):
    #     raise ValueError('Input dataframe does not have the required columns.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script entry point. You should only include columns which are either target (marked) or feature columns.')
    parser.add_argument('--input_csv', type=str, help='Input csv file.')
    parser.add_argument('--target_col', type=str, required=False, default="Name", help='Target column name.')
    parser.add_argument('--directory', type=str, help='Directory for artifacts, logs, and model, etc.')
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (default: INFO).")

    args = parser.parse_args()

    # Set log level
    logging.basicConfig(level=args.log_level)

    df = pd.read_csv(args.input_csv)

    # D checks on the input dataframe.
    do_check_on_input(df)

    # If directory does not exist.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Save the input csv file to the directory.
    input_file = os.path.join(args.directory, 'input.csv')
    df.to_csv(input_file, index=False)

    # Run training
    decision_tree = id3_algo.id3_algo(data=df,
                                      feature_cols=[col for col in df.columns if col != args.target_col],
                                      target_col=args.target_col
                                      )

    # Save the decision tree to the directory.
    decision_tree_file = os.path.join(args.directory, 'decision_tree.txt')
    with open(decision_tree_file, 'w') as f:
        f.write(json.dumps(decision_tree.to_dict(), indent=4, sort_keys=True))
