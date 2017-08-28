

def feature_extraction(data_set: iter) -> iter:
    raise NotImplementedError('Extract features from the data set.')

def init_classifier():
    raise NotImplementedError('Implement classifier.')



def make_pipeline():
    '''
    This is where we will build our pipeline.
    Pipelines are basically a chain of transformers followed by an estimator.
    The first transformer should be one (or more) methods of feature extraction.

    :return: a working pipeline.
    '''

