""" Observations are the data that we will use to train our decision classifier """

from dataclasses import dataclass, field
import enum


class FeatureType(enum.Enum):
    NUM_LEGS = "num_legs"
    COLOR = "color"
    ANIMAL = "animal"


@dataclass
class Feature:
    """ A feature is a single observation among a list of possible observations """
    name: FeatureType


@dataclass
class Animal(Feature):
    """ Animal Feature, also will serve as the target"""
    name: FeatureType.ANIMAL
    animal: str


@dataclass
class NumLegs(Feature):
    """ Number of Legs Feature"""
    name: FeatureType.NUM_LEGS
    num_legs: int


@dataclass
class Color(Feature):
    """ Color Feature"""
    name: FeatureType.COLOR
    color: str


@dataclass
class Observation:
    """ Observation is a collections of features and a label """

    features: list[Feature] = field(default_factory=list)
    label: Animal = None
