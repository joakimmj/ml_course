from help_functions import data_retriever
from tasks import sentiment_analysis


def __validate_model(clf, training_data: iter, test_data: iter, training_labels: iter, test_labels: iter,
                     average: str = 'micro', pos_label: int = 1, bitmap: bool = False):
    print('validate')


def execute_sentiment_analysis():
    print('-- Executing sentiment analysis')
    data, labels, ratings = data_retriever.load_reviews()

    print('Extracting features...')
    print('- before:\n%s' % data[0])
    feature_set = sentiment_analysis.feature_extraction(data)
    print('- after:\n%s' % feature_set[0])

    print('Splitting data...')
    training_data, test_data, training_labels, test_labels = None  # TODO: split the data
    print('- size: %d (training), %d (test)' % (len(training_labels), len(test_labels)))

    clf = sentiment_analysis.init_classifier()

    __validate_model(clf, training_data, test_data, training_labels, test_labels)

