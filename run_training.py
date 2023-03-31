import argparse
import os
import pandas as pd
import logging
from src.id3 import id3_algo
import json
from src.cli_utils import Config, do_check_on_input, coerce_dataframe


def main(args: Config):
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

    # Coerce the dataframe to the required format. Might do nothing if your inputs are all string.
    logging.info("Coercing dataframe to all string")
    df = coerce_dataframe(df, args.target_col)
    coerced_file = os.path.join(args.directory, 'coerced.csv')
    df.to_csv(coerced_file, index=False)

    # Run training
    logging.info("Running Training")
    decision_tree = id3_algo.id3_algo(data=df,
                                      feature_cols=[col for col in df.columns if col != args.target_col],
                                      target_col=args.target_col
                                      )

    # Save the decision tree to the directory.
    logging.info("Saving decision Tree")
    decision_tree_file = os.path.join(args.directory, 'decision_tree.json')
    with open(decision_tree_file, 'w') as f:
        f.write(json.dumps(decision_tree.to_dict(), indent=4, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script entry point. \
        You should only include columns which are either target (marked) or feature columns.\
        All values will be interpreted as String! TODO: support other data types! lol')

    parser.add_argument('--input_csv', type=str, help='Input csv file.')
    parser.add_argument('--target_col', type=str, required=False, default="Name", help='Target column name.')
    parser.add_argument('--directory', type=str, help='Directory for artifacts, logs, and model, etc.')
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (default: INFO).")

    args = parser.parse_args()

    # Create config object from args
    config = Config(input_csv=args.input_csv,
                    target_col=args.target_col,
                    directory=args.directory,
                    log_level=args.log_level)

    main(config)

