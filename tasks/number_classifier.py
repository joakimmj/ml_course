
def feature_extraction(data_set: iter) -> iter:
    raise NotImplementedError('Extract features from the data set.')


def split_data_set(data_set: iter, label_set: iter) -> (iter, iter, iter, iter):
    raise NotImplementedError('Split data set into training and test sets. Must return training_data, test_data, '
                              'training_labels, test_labels (in that order).')


def init_classifier():
    raise NotImplementedError('Implement classifier.')

