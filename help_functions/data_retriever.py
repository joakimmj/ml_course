import csv
import numpy as np
import os
import pickle

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


def load_reviews(reload: bool = False):
    if os.path.exists(REVIEW_DATA) and not reload:
        return pickle.load(open(REVIEW_DATA, 'rb'))

    data_set = __read_file(REVIEW_SOURCE)

    out = (data_set[:, -1], data_set[:, 0].astype(int), data_set[:, 1].astype(int))

    pickle.dump(out, open(REVIEW_DATA, 'wb'))

    return out


def load_sms(reload: bool = False):
    if os.path.exists(SMS_DATA) and not reload:
        return pickle.load(open(SMS_DATA, 'rb'))

    data_set = __read_file(SMS_SOURCE)

    out = (data_set[:, -1], data_set[:, 0].astype(int))

    pickle.dump(out, open(SMS_DATA, 'wb'))

    return out
