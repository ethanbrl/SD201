from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = sum([1 for e, a in zip(expected_results, actual_results) if e and a])
    FP = sum([1 for e, a in zip(expected_results, actual_results) if not e and a])
    FN = sum([1 for e, a in zip(expected_results, actual_results) if e and not a])

    # Calculate Precision and Recall
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)

    # Calculate F1-score using precision and recall
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score
