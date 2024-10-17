from typing import List
import pandas as pd


def compute_error(predictions: List[int], dataset: pd.DataFrame, label_column: str = 'label') -> float:
    """
    Compute the classification error.

    Args:
    - predictions: List of predicted labels.
    - dataset: DataFrame with the actual labels.
    - label_column: The name of the column with the actual labels (default is 'label').

    Returns:
    - error_rate: The error rate for the given predictions and actual labels.
    """
    total: int = len(dataset)
    errors: int = 0
    for i in range(total):
        if label_column != "":
            if predictions[i] != dataset[label_column][i]:
                errors += 1
        else:
            if predictions[i] != dataset[i]:
                errors += 1
    error_rate: float = errors / total
    return error_rate