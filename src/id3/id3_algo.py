""" ID3 Algorithm implementation """

import pandas as pd
import src.id3.id3_helper as helper
from dataclasses import dataclass, field
import pprint
import json


@dataclass
class DecisionTreeNode:
    """ Decision Tree Node """
    value: str
    children: dict[str, "DecisionTreeNode"] = field(default_factory=dict)

    # To dict
    def to_dict(self) -> dict:
        """ Converts DecisionTreeNode to JSON """
        return {
            "value": self.value,
            "children": {key: value.to_dict() for key, value in self.children.items()},
        }

    def from_dict(self, json: dict):
        """ Converts JSON to DecisionTreeNode """
        self.value = json["value"]
        self.children = {key: DecisionTreeNode.from_dict(value) for key, value in json["children"].items()}

    def print(self):
        pprint.pprint(self.to_dict(), indent=4)

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()



def id3_algo(data, feature_cols, target_col) -> DecisionTreeNode:
    """
    ID3 Algorithm implementation

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing all data
    feature_cols : List[str]
        List of feature columns
    target_col : str
        Target column

    Returns : DecisionTreeNode representing the root of the tree
    """

    # If all remaining rows have same classification, return that classification
    if len(data[target_col].unique()) == 1:
        return DecisionTreeNode(value=data[target_col].unique()[0])

    # If no more features to split on, return most common classification
    if len(feature_cols) == 0:
        return DecisionTreeNode(value=data[target_col].value_counts().idxmax())

    # Pick best feature to split on
    best_feature = helper.pick_best_feature(data, target_col, feature_cols)
    new_node = DecisionTreeNode(value=best_feature)
    for value in data[best_feature].unique():
        new_node.children[value] = id3_algo(
            data[data[best_feature] == value],
            [feature for feature in feature_cols if feature != best_feature],
            target_col,
        )
    return new_node

if __name__ == "__main__":
    # Test id3 algorithm

    df = pd.DataFrame(
        {
            "Name": ["cat", "cat", "cat", "dog", "dog", "dog", "giraffe", "zebra", "chicken", "duck", "cow", "bird"],
            "num_legs": [4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2],
            "color": ["white", "black", "black", "black", "black", "white", "white", "white", "white", "white", "white", "white"]
        }
    )

    id3_algo(df, ["num_legs", "color"], "Name").print()
    print(id3_algo(df, ["num_legs", "color"], "Name"))
