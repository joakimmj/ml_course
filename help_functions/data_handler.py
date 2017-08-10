from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer


def __format_data(words: iter):
    vocab = list({item for sublist in words for item in sublist})
    vectorizer = CountVectorizer(min_df=1, vocabulary=vocab)
    return vectorizer.fit_transform([" ".join(x) for x in words])


def split_data(feature_set: iter, labels: iter, scale_test: float = .3):
    feature_set = __format_data(feature_set)
    return model_selection.train_test_split(feature_set, labels, test_size=scale_test)
