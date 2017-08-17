from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from nltk import RegexpTokenizer
from sklearn.svm import LinearSVC
from stop_words import get_stop_words
from help_functions import validation


def feature_extraction(data_set: iter) -> iter:
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(doc.lower()) for doc in data_set]
    en_stop_words = get_stop_words('en')
    words = [[i for i in token_list if i not in en_stop_words] for token_list in tokens]
    vocab = list({item for sublist in words for item in sublist})
    vectorizer = CountVectorizer(min_df=1, vocabulary=vocab)
    return vectorizer.fit_transform([" ".join(x) for x in words])


def split_data_set(data_set, label_set):
    return model_selection.train_test_split(data_set, label_set, test_size=.3)


def classifier():
    return LinearSVC()


if __name__ == '__main__':
    validation.validate_spam_filter(feature_extraction, split_data_set, classifier)

# TODO : test if main needed
