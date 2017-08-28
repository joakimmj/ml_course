import numpy as np
from sklearn import metrics, utils
from help_functions import data_retriever, bitmap_handler
from tasks import spam_filter, number_classifier


def __validate_model(clf, test_src: iter, training_data: iter, test_data: iter, training_labels: iter,
                     test_labels: iter, average: str = 'micro', pos_label: int = 1, bitmap: bool = False):
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
        bitmap_handler.compare_wrong_results(test_src, predictions, test_labels)
    else:
        for i, label in enumerate(test_labels):
            if predictions[i] != label:
                print('Label: %s Predicted: %s Msg: %s' % (label, predictions[i], test_src[i]))


def __split_data_set(data_set: np.ndarray, test_size: float):
    split = int(data_set.shape[0] * test_size)
    return data_set[:split], data_set[split:]


def __execute(data_loader, feature_extractor, classifier, test_size: float, bitmap: bool):
    data, labels = data_loader()
    data, labels = utils.shuffle(data, labels)

    print('Extracting features...')
    print('- before:\n%s' % data[0])
    feature_set = feature_extractor(data)
    print('- after:\n%s' % feature_set[0])

    print('Splitting data...')
    test_data, training_data = __split_data_set(feature_set, test_size)
    test_labels, training_labels = __split_data_set(labels, test_size)
    test_src, _ = __split_data_set(data, test_size)
    print('- size: %d (training), %d (test)' % (len(training_labels), len(test_labels)))

    clf = classifier()

    __validate_model(clf, test_src, training_data, test_data, training_labels, test_labels, bitmap=bitmap)


def execute_spam_filter(test_size: float = .3):
    print('-- Executing spam filter')
    __execute(data_retriever.load_sms, spam_filter.feature_extraction,
              spam_filter.init_classifier, test_size, bitmap=False)


def execute_number_classifier(test_size: float = .3):
    print('-- Executing number classification')
    __execute(data_retriever.load_mnist, number_classifier.feature_extraction,
              number_classifier.init_classifier, test_size, bitmap=True)
