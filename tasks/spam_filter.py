from sklearn.base import ClassifierMixin, TransformerMixin
import numpy as np


def feature_extraction(data_set: iter) -> iter:
    raise NotImplementedError('Extract features from the data set.')


def init_classifier():
    raise NotImplementedError('Implement classifier.')


class SMSFeatureExtractor(TransformerMixin):
    """
    We are going to make a Transformer.
    Transformers are generally used for two things:
        1. Extract features from raw data (such as a list of smses).
        2. Transform a feature set into another feature set.
    """

    def fit(self, documents: iter, *others):
        """
        The goal of this method is to do review the data and prepare for any transformation.
        For this task we are going to make a simple word bag model, so we have to store the amount
        of available words.
        :param documents: A list of text messages.
        :param *others: Stuff other scikit-learn modules might tack on, that we will ignore.
        :return: The Transformer itself. This allows for method-chaining.
        """
        raise NotImplementedError('You should store all the distinct words here.')
        return self

    def transform(self, documents, *others):
        """
        This method is where we do the feature extraction. It is called transform because we are
        transforming the data from one representation to another.
        :param documents:  A list of text messages.
        :param y: Stuff other scikit-learn modules might tack on, that we will ignore.
        :return: An NxM matrix where N is the amount of text messages and M is the amount of features (words).
        """
        raise NotImplementedError('Return the features. See docstring for more details.')


class ExpectedValueClassifier(ClassifierMixin):
    """
    The stupidest estimator I could think of.
    Finds the most commonly occurring label and returns that for all the samples.
    """

    def __init__(self):
        self.most_frequent_class = None

    def fit(self, X, y):
        """
        This is where we train the model on the data.
        Here we are just going to count the number of occurrences of each label, storing the most frequent.
        :param X: The features. We ignore those here.
        :param y: The labels, which we will be counting.
        :return:
        """
        minimum = np.min(y)
        offset = 0
        if minimum < 0:
            offset = np.abs(minimum)
        histogram = np.bincount(y + offset)
        self.most_frequent_class = histogram.argmax() - offset

        return self

    def predict(self, X):
        """
        This method predicts labels for the given features based on its training.
        :param X: The labels of the data we want to predict.
        :return: An vector of labels.
        """
        return np.asarray([self.most_frequent_class] * X.shape[0])
