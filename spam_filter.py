from help_functions import validation


def feature_extraction(data_set: iter) -> iter:
    raise NotImplementedError('Extract features from the data set.')


def split_data_set(data_set, label_set):
    raise NotImplementedError('Split data set into training and test sets. Must return training_data, test_data, '
                              'training_labels, test_labels (in that order).')


def classifier():
    raise NotImplementedError('Implement classifier. Must return a Scikit-learn estimator.')


if __name__ == '__main__':
    validation.validate_spam_filter(feature_extraction, split_data_set, classifier)
