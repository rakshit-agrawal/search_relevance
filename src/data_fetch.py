import os
from pprint import pprint

__author__ = 'rakshit'

import csv
import pickle


def csv_read(filename, train_dict={}):
    csvfile = open(filename, 'rb')
    taskreader = csv.reader(csvfile, delimiter=',')
    for row in taskreader:
        for i, v in enumerate(LABELS):
            if i == 0:
                ident = row[i]
                train_dict[ident] = {}
            else:
                train_dict[ident][v] = row[i]

    return train_dict


def get_file(filename, picklename):
    # Check for pickle of the given file and return dict
    if os.path.isfile(picklename):
        ret_dict = pickle.load(open(picklename, "rb"))
        return ret_dict
    else:
        ret_dict = csv_read(filename)
        try:
            ret_dict.pop('id')
        except:
            pass
        pickle.dump(ret_dict, open(picklename, "wb"))
        return ret_dict


def generate_file_names(base_name, type):
    filename = "../data/" + base_name + "." + type
    picklename = base_name + ".p"

    return filename, picklename

if __name__ == "__main__":
    FILENAME = "../data/train.csv"
    BASE_TRAIN = "train"
    FILE_TYPE = "csv"
    LABELS = ["id", "query", "product_title", "product_description", "median_relevance", "relevance_variance"]

    filename, picklename = generate_file_names(BASE_TRAIN,FILE_TYPE)
    train_dict = get_file(filename, picklename)

    pprint(train_dict)