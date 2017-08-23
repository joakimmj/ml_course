from sklearn import metrics
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

    # for i, prediction in enumerate(predictions):
    #     if prediction != test_labels[i]:
    #         print('wrong!!')
    #         print(test_data[i])

    # TODO: print wrongly classified 


def validate_spam_filter(feature_extraction, split_data_set, classifier):
    data, labels = data_retriever.load_sms()

    feature_set = feature_extraction(data)
    training_data, test_data, training_labels, test_labels = split_data_set(feature_set, labels)
    clf = classifier()

    __validate_model(clf, training_data, test_data, training_labels, test_labels)


def validate_sentiment_analysis(feature_extraction, split_data_set, classifier):
    data, labels, ratings = data_retriever.load_reviews()

    feature_set = feature_extraction(data)
    training_data, test_data, training_labels, test_labels = split_data_set(feature_set, labels, ratings)
    clf = classifier()

    __validate_model(clf, training_data, test_data, training_labels, test_labels)
