from typing import List

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.points = PointSet(features, labels, types)

        self.left_child = None
        self.right_child = None

        # For Q9
        self.min_split_points = min_split_points

        # We test if it is a Leaf
        if h != 0 and len(self.points.labels) > self.min_split_points:
            self.split_index, _ = self.points.get_best_gain(self.min_split_points)

            # Check if our split index is not None
            if self.split_index is not None:
                # For REAL features
                if self.points.types[self.points.best_feature_index] == FeaturesTypes.REAL:
                    left = self.points.features[:, self.split_index] <= self.points.best_value
                    right = self.points.features[:, self.split_index] > self.points.best_value 
            
                # For BOOLEAN and CATEGORICAL features
                else:
                    left = self.points.features[:, self.split_index] == self.points.best_value
                    right = self.points.features[:, self.split_index] != self.points.best_value 

                left_set = PointSet(self.points.features[left], self.points.labels[left], types)
                right_set = PointSet(self.points.features[right], self.points.labels[right], types)
            
                # We test if the sons are not leaves
                if (left_set.features.shape[0] > 0 and right_set.features.shape[0] > 0):
                    self.left_child = Tree(left_set.features.tolist(), left_set.labels, types, h=h - 1, min_split_points=min_split_points)
                    self.right_child = Tree(right_set.features.tolist(), right_set.labels, types, h=h - 1, min_split_points=min_split_points)

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point
&
        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        # Before deciding we check that the left and right sons are not Empty
        if self.left_child is not None and self.right_child is not None:

            # REAL Feature
            if self.points.types[self.points.best_feature_index] == FeaturesTypes.REAL:
                if features[self.split_index] > self.points.best_value:
                    return self.right_child.decide(features)
                else:
                    return self.left_child.decide(features)

            # BOOLEAN or CATEGORICAL Feature
            else:  
                if features[self.split_index] != self.points.best_value:
                    return self.right_child.decide(features)
                else:
                    return self.left_child.decide(features)
        else:
            return bool(self.points.labels.sum() > len(self.points.labels) / 2)