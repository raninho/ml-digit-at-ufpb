from typing import List
from sklearn.metrics import accuracy_score

def compute_error(y_train: List[int], y_train_pred: List[int]):
    return 1 - accuracy_score(y_train, y_train_pred)
