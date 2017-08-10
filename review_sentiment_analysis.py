from nltk import RegexpTokenizer
from sklearn.tree import DecisionTreeClassifier
from stop_words import get_stop_words
from help_functions import data_retriever, clf_handler, data_handler


def transform_data(data_set):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(doc.lower()) for doc in data_set]
    en_stop_words = get_stop_words('en')
    out = [[i for i in token_list if not i in en_stop_words] for token_list in tokens]
    return out

print('Retrieve data')
data, labels = data_retriever.load_reviews()

print('Format data for training')
feature_set = transform_data(data)

print('Split data into training and test sets')
scale_test = .2
training_data, test_data, training_labels, test_labels = data_handler.split_data(feature_set, labels, scale_test)

clf = DecisionTreeClassifier()

print('Training classifier...')
clf.fit(training_data, training_labels)

print('Review of classifier')
clf_handler.review_clf(clf, test_data, test_labels)
