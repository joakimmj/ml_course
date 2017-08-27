import csv
import numpy as np
import os
import pickle
from sklearn import datasets, utils

REVIEW_SOURCE = './files/review_data/review_source.csv'
REVIEW_DATA = './files/review_data/review_data.sav'
SMS_SOURCE = 'files/spam_data/sms_source.csv'
SMS_DATA = 'files/spam_data/sms_data.sav'


def __read_file(source):
    data_set = []
    with open(source, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader.__next__()

        for row in csv_reader:
            data_set.append(row)

    return np.array(data_set)


def __load_file(source_location: str, save_location: str, reload: bool):
    print('Retrieving data...')

    if os.path.exists(save_location) and not reload:
        return pickle.load(open(save_location, 'rb'))

    data_set = __read_file(source_location)
    pickle.dump(data_set, open(save_location, 'wb'))
    return data_set


def load_reviews(reload: bool = False):
    data_set = __load_file(REVIEW_SOURCE, REVIEW_DATA, reload)
    return data_set[:, -1], data_set[:, 0].astype(int), data_set[:, 1].astype(int)


def load_sms(reload: bool = False):
    data_set = __load_file(SMS_SOURCE, SMS_DATA, reload)
    return data_set[:, -1], data_set[:, 0].astype(int)


def load_mnist():
    mnist = datasets.fetch_mldata('MNIST original', data_home='./files')
    print("Fetched %d bitmaps." % len(mnist.target))

    print("Shuffle data set")
    mnist.data, mnist.target = utils.shuffle(mnist.data, mnist.target)

    return mnist.data, mnist.target
