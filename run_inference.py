""" Script for running inference with trained decision tree. """

import argparse
import json
import logging
import os
from src.id3 import id3_algo
import pandas as pd

from run_training import Config, coerce_dataframe, do_check_on_input


def main(config: Config):
    """ Main function for running inference. """

    # Set log level
    logging.basicConfig(level=config.log_level)

    # make sure the directory exists
    if not os.path.exists(config.directory):
        os.makedirs(config.directory)

    # Load test dataframe
    test_df = pd.read_csv(config.input_csv)

    # Do checks on the input dataframe.
    do_check_on_input(test_df)

    # Coerce the dataframe to the required format. Might do nothing if your inputs are all string.
    logging.info("Coercing dataframe to all string")
    test_df = coerce_dataframe(test_df, config.target_col)

    # Load the decision tree
    decision_tree_file = os.path.join(config.directory, 'decision_tree.json')
    with open(decision_tree_file, 'r') as f:
        decision_tree: id3_algo.DecisionTreeNode = id3_algo.DecisionTreeNode.from_dict(json.load(f))

    test_df.loc[:, "prediction"] = test_df.apply(lambda row: id3_algo.use_tree(row, decision_tree), axis=1)
    logging.info(test_df.head())

    # Save dataframe to output directory
    output_file = os.path.join(config.directory, 'output.csv')
    test_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inference script entry point. \
            Only feature columns used in trained decision tree will be used..')

    parser.add_argument('--test_csv', type=str, help='Test csv file.')
    parser.add_argument('--do_eval', action="store_true", help='Run evaluation with target column')
    parser.add_argument('--model_dir', type=str, help='Directory of model to use for inference.')
    parser.add_argument("--out_dir", type=str, help="Directory for output.")
    parser.add_argument("--target_col", type=str, default="Name", help="Target col, used for formatting and evaluation.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (default: INFO).")

    args = parser.parse_args()

    # Create config object from args
    input_config = Config(input_csv=args.test_csv,
                          target_col=args.target_col,
                          directory=args.model_dir,
                          log_level=args.log_level,
                          do_eval=False)

    main(input_config)
