from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.best_value = None
        self.best_feature_index = None
        self.best_gini_gain = None
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        total_points = self.features.shape[0]

        if total_points == 0:
            return 0

        true = np.count_nonzero(self.labels)
        false = total_points - true

        prob_true = true / total_points
        prob_false = false / total_points

        gini_score = 1 - (prob_true ** 2) - (prob_false ** 2)

        return gini_score

    def get_best_gain(self, min_split=1) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        # Intitial Gini Gain
        first_gini = self.get_gini()

        # We iterate through the features
        for feature_index in range(self.features.shape[1]):

            values = []
            
            # check the Feature Type and raise an Error if it's an unknown one
            if self.types[feature_index] == FeaturesTypes.BOOLEAN:
                feature_type = FeaturesTypes.BOOLEAN
                values = [0,1]

            elif self.types[feature_index] == FeaturesTypes.CLASSES:
                feature_type = FeaturesTypes.CLASSES
                values = np.unique(self.features[:, feature_index])

            elif self.types[feature_index] == FeaturesTypes.REAL:
                feature_type = FeaturesTypes.REAL
                values = sorted(np.unique(self.features[:, feature_index]))
            
            else:
                raise ValueError("Unknown feature type. Here are the feature types :", self.types)

            # We iterate through the values of the column
            for value in values:

                if feature_type == FeaturesTypes.BOOLEAN:

                    left_split = self.features[:, feature_index] == value
                    right_split = self.features[:, feature_index] != value

                elif feature_type == FeaturesTypes.CLASSES:

                    left_split = self.features[:, feature_index] == value
                    right_split = ~left_split

                elif feature_type == FeaturesTypes.REAL:

                    temp_left_split = self.features[:, feature_index] <= value
                    temp_right_split = self.features[:, feature_index] > value

                    if self.features[temp_right_split, feature_index].size > 0:
                        max_left = np.max(self.features[temp_left_split, feature_index])
                        min_right = np.min(self.features[temp_right_split, feature_index])

                        value = (min_right + max_left)/2
                    
                    left_split = self.features[:, feature_index] <= value
                    right_split = self.features[:, feature_index] > value

                # Compute the Gini scores for the left and right subsets
                left_set = PointSet(self.features[left_split], self.labels[left_split], self.types)
                right_set = PointSet(self.features[right_split], self.labels[right_split], self.types)

                # Get the gain for left and right sons
                left_gini = left_set.get_gini()
                right_gini = right_set.get_gini()

                dataset_size = self.labels.size
                left_size = left_set.labels.size
                right_size = right_set.labels.size

                # Compute the Total Gini
                gini_gain = first_gini - (left_size / dataset_size) * left_gini - (right_size / dataset_size) * right_gini

                # Update the best gain and give which value is the best (Threshold) (check the size of the two sons for Q9)
                if (self.best_gini_gain is None or gini_gain > self.best_gini_gain) and (left_size >= min_split and right_size >= min_split):
                    self.best_gini_gain = gini_gain
                    self.best_feature_index = feature_index
                    self.best_value = value

        return self.best_feature_index, self.best_gini_gain
    
    def get_best_threshold(self) -> float:

        # Depending on the type we return the threshold value or None and raise an exception if the function get_best_gain() has not been ran
        if self.best_value is not None:

            if self.types[self.best_feature_index] == FeaturesTypes.REAL:
                return self.best_value
            
            elif self.types[self.best_feature_index] == FeaturesTypes.CLASSES:
                return self.best_value
            
            elif self.types[self.best_feature_index] == FeaturesTypes.BOOLEAN:
                return None
            else:
                raise ValueError("Unknown feature type. Here are the feature types :", self.types)
        else:
            raise Exception("use the function get_best_gain() before")