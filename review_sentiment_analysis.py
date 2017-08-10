from help_functions import data_retriever, clf_handler, data_handler


def transform_data(data_set):
    # TODO : transform the data
    return data_set

print('Retrieving data...')
data, labels, ratings = data_retriever.load_reviews()

print('Formatting data...')
feature_set = transform_data(data)

print('Splitting data into training and test sets...')
scale_test = .0  # TODO: choose size of test set
training_data, test_data, training_labels, test_labels = data_handler.split_data(feature_set, labels, scale_test)

clf = None  # TODO: choose classifier

print('Training classifier...')
clf.fit(training_data, training_labels)

print('Testing classifier...')
clf_handler.review_clf(clf, test_data, test_labels)
