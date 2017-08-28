import numpy as np
from sklearn import metrics, model_selection
from help_functions import data_retriever, bitmap_handler
from tasks import spam_filter, number_classifier


def __validate_model(clf, training_data: iter, test_data: iter, training_labels: iter, test_labels: iter,
                     average: str = 'micro', pos_label: int = 1, bitmap: bool = False):
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

    if bitmap:
        bitmap_handler.compare_wrong_results(test_data, predictions, test_labels)


def __execute(data_loader, feature_extractor, classifier):
    data, labels = data_loader()

    print('Extracting features...')
    print('- before:\n%s' % data[0])
    feature_set = feature_extractor(data)
    print('- after:\n%s' % feature_set[0])

    print('Splitting data...')
    training_data, test_data, training_labels, test_labels = model_selection.train_test_split(feature_set, labels, test_size=.3)  # TODO: split data and show errors
    print('- size: %d (training), %d (test)' % (len(training_labels), len(test_labels)))

    clf = classifier()

    __validate_model(clf, training_data, test_data, training_labels, test_labels)


def execute_spam_filter():
    print('-- Executing spam filter')
    __execute(data_retriever.load_sms, spam_filter.feature_extraction, spam_filter.init_classifier)


def execute_number_classifier():
    print('-- Executing number classification')
    __execute(data_retriever.load_mnist, number_classifier.feature_extraction, number_classifier.init_classifier)
