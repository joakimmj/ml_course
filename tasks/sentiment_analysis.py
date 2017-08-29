from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import LatentDirichletAllocation


def make_pipeline():
    '''
    This is where we will build our pipeline.
    Pipelines are basically a chain of transformers followed by an estimator.
    The first transformer should be one (or more) methods of feature extraction.

    :return: a working pipeline.
    '''
    #TODO remove pipeline steps
    pipeline_steps = [
        ('cv', CountVectorizer()),
        ('svr', LinearSVC())
    ]

    return Pipeline(steps=pipeline_steps)




