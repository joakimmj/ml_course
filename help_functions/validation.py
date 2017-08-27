from sklearn import metrics
import sentiment_analysis
import spam_filter
from help_functions import data_retriever
import numpy as np


def __validate_model(clf, training_data: iter, test_data: iter, training_labels: iter,
                     test_labels: iter, average='micro', pos_label=1):
    multi_class = len(np.unique(training_labels, return_counts=True)[0]) > 2

    print('Training classifier (%s)...' % ('multi class' if multi_class else 'binary'))
    clf.fit(training_data, training_labels)

    print('Testing classifier...')
    predictions = clf.predict(test_data)

    classification_report = metrics.classification_report(test_labels, predictions)
    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)

    if multi_class:
        f1_score = metrics.f1_score(test_labels, predictions, average=average)
        precision = metrics.precision_score(test_labels, predictions, average=average)
        recall = metrics.recall_score(test_labels, predictions, average=average)
    else:
        f1_score = metrics.f1_score(test_labels, predictions, pos_label=pos_label)
        precision = metrics.precision_score(test_labels, predictions, pos_label=pos_label)
        recall = metrics.recall_score(test_labels, predictions, pos_label=pos_label)

    print('Classification report:\n%s\n\nConfusion matrix:\n%s\n\nF-score: %.2f\nPrecision: %.2f\nRecall: %.2f'
          % (classification_report, confusion_matrix, f1_score, precision, recall))


def execute_spam_filter():
    print('-- Executing spam filter')
    data, labels = data_retriever.load_sms()

    feature_set = spam_filter.feature_extraction(data)
    training_data, test_data, training_labels, test_labels = spam_filter.split_data_set(feature_set, labels)
    clf = spam_filter.init_classifier()

    __validate_model(clf, training_data, test_data, training_labels, test_labels)


def execute_sentiment_analysis():
    print('-- Executing sentiment analysis')
    data, labels, ratings = data_retriever.load_reviews()

    feature_set = sentiment_analysis.feature_extraction(data)
    training_data, test_data, training_labels, test_labels = sentiment_analysis.split_data_set(feature_set, labels, ratings)
    clf = sentiment_analysis.init_classifier()

    __validate_model(clf, training_data, test_data, training_labels, test_labels)
