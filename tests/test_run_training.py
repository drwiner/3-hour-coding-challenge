import pytest
import pathlib

from src.cli_utils import Config
from run_training import main


@pytest.mark.parametrize(
    "input_csv, target_col, directory, log_level",
    [
        ("resources/animals.csv", "Name", "test_animals", "INFO"),
        ("resources/test.csv", "Name", "test_debug", "DEBUG"),
        ("resources/animals.csv", "Number of legs", "test_legs", "WARNING"),
        ("resources/animals.csv", "Color", "test_color", "INFO"),
    ],
)
def test_main(input_csv, target_col, directory, log_level):
    """ Test that inputs work without error """
    args = Config(input_csv, target_col, directory, log_level)
    parent_dir = pathlib.Path(__file__).parent.resolve()
    args.input_csv = f"{parent_dir}/../{args.input_csv}"
    main(args)


def test_end_to_end(input_csv, target_col, directory, log_level, test_csv, expected_output):
    """ Test train and inference + eval in end to end pipeline on toy examples. """
    pass


if __name__ == "__main__":
    # Just for testing purposes
    test_main("/Users/drw/siamak-test/resources/animals.csv", "Name", "/Users/drw/siamak-test/test_animals", "INFO")