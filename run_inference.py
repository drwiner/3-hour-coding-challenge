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

    if not os.path.exists(config.inference_directory):
        os.makedirs(config.inference_directory)

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
    output_file = os.path.join(config.inference_directory, 'output.csv')
    test_df.to_csv(output_file, index=False)

    if config.do_eval:
        # Evaluate the model. These are probably wrong!
        logging.info("Evaluating model")
        accuracy = test_df.loc[test_df[config.target_col] == test_df["prediction"]].shape[0] / test_df.shape[0]
        logging.info(f"Accuracy: {accuracy}")

        # For each value of target col, there is precision, recall, and f1
        target_values = test_df[config.target_col].unique()
        for target_value in target_values:
            true_positives = test_df.loc[(test_df[config.target_col] == target_value) & (test_df["prediction"] == target_value)].shape[0]
            false_positives = test_df.loc[(test_df[config.target_col] != target_value) & (test_df["prediction"] == target_value)].shape[0]
            false_negatives = test_df.loc[(test_df[config.target_col] == target_value) & (test_df["prediction"] != target_value)].shape[0]

            if true_positives + false_positives == 0:
                precision = 0
            else:
                precision = true_positives / (true_positives + false_positives)

            if (true_positives + false_negatives) == 0:
                recall = 0
            else:
                recall = true_positives / (true_positives + false_negatives)

            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            logging.info(f"Target: {target_value}\tPrecision: {precision}, Recall: {recall}, F1: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inference script entry point. \
            Only feature columns used in trained decision tree will be used..')

    parser.add_argument('--test_csv', type=str, help='Test csv file.')
    parser.add_argument('--do_eval', action="store_true", help='Run evaluation with target column')
    parser.add_argument('--model_dir', type=str, help='Directory of model to use for inference.')
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for output.")
    parser.add_argument("--target_col", type=str, default="Name", help="Target col, used for formatting and evaluation.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (default: INFO).")

    args = parser.parse_args()

    if not args.out_dir:
        # By default, we'll just save it in the model directory
        args.out_dir = args.model_dir

    # Create config object from args
    input_config = Config(input_csv=args.test_csv,
                          target_col=args.target_col,
                          directory=args.model_dir,
                          inference_directory=args.out_dir,
                          log_level=args.log_level,
                          do_eval=args.do_eval)

    main(input_config)
