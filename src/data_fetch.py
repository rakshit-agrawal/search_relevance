from math import log
import os
import csv
import pickle
from pprint import pprint
from nltk import RegexpTokenizer, FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = 'rakshit'




def get_tokens(dict_element):
    # Remove stop words from data and perform initial
    # cleanup for feature extraction

    query = dict_element['query']
    desc = dict_element['product_description']
    title = dict_element['product_title']
    stop = stopwords.words('english')

    pattern = r'''(?x)               # set flag to allow verbose regexps
          ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
          | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
          | \w+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
          | @((\w)+([-']\w+))*
          | [+/\-@&*]        # special characters with meanings
        '''

    #pattern = r'[+/\-@&*#](\w+)|(\w+)'
    tokenizer = RegexpTokenizer(pattern)



    #tokenizer = RegexpTokenizer(r'\w+')
    query_tokens = tokenizer.tokenize(query)
    query_tokens = map(lambda x:x.lower(),query_tokens)
    desc_tokens = tokenizer.tokenize(desc)
    desc_tokens = [x.lower() for x in desc_tokens if x.lower() not in stop]
    title_tokens = tokenizer.tokenize(title)
    title_tokens = [x.lower() for x in title_tokens if x.lower() not in stop]

    return query_tokens, title_tokens, desc_tokens


def csv_read(filename, labels, build_dict={}):
    csvfile = open(filename, 'rb')
    taskreader = csv.reader(csvfile, delimiter=',')
    for row in taskreader:
        for i, v in enumerate(labels):
            if i == 0:
                ident = row[i]
                build_dict[ident] = {}
            else:
                build_dict[ident][v] = row[i]

    return build_dict


def get_file(filename, picklename, labels):
    # Check for pickle of the given file and return dict
    if os.path.isfile(picklename):
        ret_dict = pickle.load(open(picklename, "rb"))
        return ret_dict
    else:
        ret_dict = csv_read(filename, labels)
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


def main_ops(data_dict, t_map_list=[], relevance_list=[], d_map_list=[], variance_list=[]):
    # Main operations on the dictionary elements
    dimx = len(data_dict)
    dimy = 3
    ctr = 0
    splitter = 0.20*dimx
    data_mat = np.empty((dimx,dimy))
    target = np.empty((dimx,), dtype=np.int)


    for k,v in data_dict.iteritems():
        t_map = 0.0
        d_map = 0.0

        qt, tt, dt = get_tokens(v)
        for token in qt:
            if token in tt:
                t_map+=1
            if token in dt:
                d_map+=1

        t_map = 0.0 + t_map/len(qt)
        d_map = 0.0 + d_map/len(qt)

        data_mat[ctr][0] = t_map
        data_mat[ctr][1] = d_map
        data_mat[ctr][2] = v["relevance_variance"]

        target[ctr] = v["median_relevance"]

        t_map_list.append(t_map)
        d_map_list.append(d_map)

        relevance_list.append(v["median_relevance"])
        variance_list.append(v["relevance_variance"])


        #print t_map, v["median_relevance"]
        ctr+=1
            #print qt
            #print tt
            #print t_map, d_map
            #print v["median_relevance"]
    #plt.show(plt.plot(t_map_list, relevance_list))
    #plt.scatter(t_map_list, relevance_list)
    #plt.show() # this plot shows no significant relation between a relevance and mapping of keywords with title
    train_data = data_mat[:-splitter]
    test_data = data_mat[-splitter:]
    train_target = target[:-splitter]
    test_target = target[-splitter:]
    #print train_data

    clf = svm.SVC()
    fit_rtn = clf.set_params(kernel='rbf').fit(train_data, train_target)
    print fit_rtn

    correct = wrong = 0
    results=set()
    for i,v in enumerate(test_data):
        res = clf.predict(v)
        label = test_target[i]
        results.add(res[0])
        if res[0]==label:
            correct+=1
        else:
            wrong+=1
        print "Predicted = {}, True = {}".format(res,label)
        print correct
        print wrong
        print results
    success = 0.0 + correct/(0.0 + correct + wrong)
    print success


def tf_measure(word_tokens, query_tokens, N):
    tfscore = 0.0

    freq = FreqDist(word_tokens)
    try:
        wf = max(freq.values())
    except:
        wf = 0.0

    for token in query_tokens:
        try:
            tf = freq[token]
            tf = 1.0 + log(tf)
            #tfscore = 0.5 + (0.5 * (0.0 + tf))/(0.0 + wf)
        except:
            tf = 0.0

        tfscore+=tf



    # IDF measures



    #print tfscore
    return tfscore


def tfidf_ops(data_dict, t_map_list=[], relevance_list=[], d_map_list=[], variance_list=[]):
    # TF-IDF operations

    dimx = len(data_dict)
    dimy = 4
    ctr = 0
    splitter = 0.20*dimx
    data_mat = np.empty((dimx,dimy))
    target = np.empty((dimx,), dtype=np.int)
    stop = stopwords.words('english')


    for k,v in data_dict.iteritems():
        t_map = 0.0
        d_map = 0.0

        qt, tt, dt = get_tokens(v)

        for token in qt:
            if token in tt:
                t_map+=1
            if token in dt:
                d_map+=1


        dtfscore = tf_measure(dt,qt,dimx)
        ttfscore = tf_measure(tt,qt,dimx)

        data_mat[ctr][0] = dtfscore
        data_mat[ctr][1] = d_map
        data_mat[ctr][2] = ttfscore
        data_mat[ctr][3] = t_map

        target[ctr] = v["median_relevance"]

        t_map_list.append(t_map)
        d_map_list.append(d_map)

        relevance_list.append(v["median_relevance"])
        variance_list.append(v["relevance_variance"])


        #print t_map, v["median_relevance"]
        ctr+=1
            #print qt
            #print tt
            #print t_map, d_map
            #print v["median_relevance"]
    #plt.show(plt.plot(t_map_list, relevance_list))
    #plt.scatter(t_map_list, relevance_list)
    #plt.show() # this plot shows no significant relation between a relevance and mapping of keywords with title
    train_data = data_mat[:-splitter]
    test_data = data_mat[-splitter:]
    train_target = target[:-splitter]
    test_target = target[-splitter:]
    #print train_data

    clf = svm.SVC(gamma=0.001, C=100.)
    fit_rtn = clf.fit(train_data, train_target)
    print fit_rtn

 #   return clf, test_data, test_target

#def predictions(clf, test_data, test_target):

    correct = wrong = 0
    results=set()
    for i,v in enumerate(test_data):
        res = clf.predict(v)
        label = test_target[i]
        results.add(res[0])
        if res[0]==label:
            correct+=1
        else:
            wrong+=1
        print "Predicted = {}, True = {}".format(res,label)
        print correct
        print wrong
        print results
    success = 0.0 + correct/(0.0 + correct + wrong)
    print success


if __name__ == "__main__":
    FILENAME = "../data/train.csv"
    BASE_TRAIN = "train"
    BASE_TEST = "test"
    FILE_TYPE = "csv"
    LABELS = ["id", "query", "product_title", "product_description", "median_relevance", "relevance_variance"]

    filename, picklename = generate_file_names(BASE_TRAIN,FILE_TYPE)
    train_dict = get_file(filename, picklename, LABELS)

    filename, picklename = generate_file_names(BASE_TEST,FILE_TYPE)
    test_dict = get_file(filename, picklename, LABELS[:3])

    #pprint(train_dict)

    #main_ops(train_dict)
    #clf, test_data, test_target = \
    tfidf_ops(train_dict)
   # predictions(clf, test_data, test_target)