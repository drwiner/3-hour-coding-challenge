""" Helper script for CLI."""
from dataclasses import dataclass
import pandas as pd

@dataclass
class Config:
    """Config class to hold all the configuration parameters."""
    input_csv: str
    target_col: str
    directory: str
    inference_directory: str
    log_level: str
    do_eval: bool = False


def do_check_on_input(input_df: pd.DataFrame) -> None:
    """Do some checks on the input dataframe

    Args:
        input_df (pd.DataFrame): Input whose columns we have to check.

    Raises:
        ValueError: If the input dataframe does not have the required columns or is empty.
    """
    if input_df.empty:
        raise ValueError('Input dataframe is empty.')


def coerce_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Coerce the input dataframe to the required format.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Target column.

    Returns:
        pd.DataFrame: Coerced dataframe.
    """
    # Convert all the columns to string.
    df = df.astype(str)

    # Convert the target column to lowercase.
    df[target_col] = df[target_col].str.lower()

    return df
