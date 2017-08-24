
def feature_extraction(data_set: iter, rating_set: iter = None) -> iter:
    raise NotImplementedError('Extract features from the data set.')


def split_data_set(data_set, label_set):
    raise NotImplementedError('Split data set into training and test set.')


def classifier():
    raise NotImplementedError('Implement classifier.')
