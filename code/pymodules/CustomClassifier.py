import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class NullClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        """
        Called when initializing the classifier
        """
        self.class_prob = None
        self._fitted = False

    def fit(self, X, y):
        """
        Based on frequency of y, fit
        """
        # Check that X and y have correct shape
        check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.class_prob = y.value_counts()/len(y)
        self._fitted = True

        return self

    def decision_function(self, X):
        D = [np.random.choice(self.classes_, p=self.class_prob, replace=True) for _ in range(len(X))]
        return np.array(D)

    def predict(self, X):
        check_is_fitted(self, ['_fitted'])
        # Input validation
        check_array(X)
        D = self.decision_function(X)
        return D

