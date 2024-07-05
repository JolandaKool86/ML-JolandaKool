import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Accuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y, yhat):
        return (np.argmax(yhat, axis=1) == y).sum() / len(yhat)


class F1Score:
    def __init__(self, average: str):
        self.average = average

    def __repr__(self) -> str:
        return f"F1score{(self.average)}"

    def __call__(self, y, yhat):
        return f1_score(
            y, np.argmax(yhat, axis=1), average=self.average, zero_division=np.nan
        )


class Recall:
    def __init__(self, average: str):
        self.average = average

    def __repr__(self) -> str:
        return f"Recall{(self.average)}"

    def __call__(self, y, yhat):
        return recall_score(
            y, np.argmax(yhat, axis=1), average=self.average, zero_division=np.nan
        )


class Precision:
    def __init__(self, average: str):
        self.average = average

    def __repr__(self) -> str:
        return f"Precision{(self.average)}"

    def __call__(self, y, yhat):
        return precision_score(
            y, np.argmax(yhat, axis=1), average=self.average, zero_division=np.nan
        )


class ThresholdedRecall:
    def __init__(self, threshold: float = 0.2, average: str = "micro"):
        self.threshold = threshold
        self.average = average

    def __repr__(self) -> str:
        return f"ThresholdedRecall_threshold_{self.threshold}_average_{self.average}"

    def __call__(self, y, yhat):
        yhat_thresholded = (yhat >= self.threshold).astype(int)

        if y.ndim == 1:
            # Binary classification case
            y_true = y
            y_pred = yhat_thresholded[
                :, 0
            ]  # Assuming binary classification has shape (n_samples, 1)
        else:
            # Multilabel case
            y_true = y
            y_pred = yhat_thresholded

        return recall_score(y_true, y_pred, average=self.average, zero_division=np.nan)
