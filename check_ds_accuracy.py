"""
author: audreyc
Last Update: 05/03/16
"""

import sys
import os.path
import DS_type_mapping
# import gen_files
import create_train_files_by_company
import audrey_model
import numpy as np
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
import time

company_list = ['p0000745jr8u', 'p0003884x7lt', 'p0043611aoji', 'p0039557fvbf',
                'p0090051h2oq', 'p0006679vz2s', 'p00425z4gu', 'p0009976ed7k',
                'p0005859huep', 'p0089280mxzg', 'p0079383unbs', 'p00521862bvn',
                'p0002406hlem', 'p0002233vdsm', 'p0096373arhm', 'p0039096va39',
                'p0046598vm29', 'p0036394dcvf', 'p0081759udz2', 'p0079434mdig',
                'p0014655asb9', 'p0043883wrzh']

if __name__ == '__main__':
    create_train_files_by_company.parse_file('all')
    x_train = pickle.load(open('intermediate/all_x_test.pkl', 'rb'))
    y_train = pickle.load(open('intermediate/all_y_testDS.pkl', 'rb'))
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(x_train)
    print("Training set, num features: " + str(X.shape))
    clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
    clf.fit(X, y_train)
    for company in reversed(company_list):
        print("-------- ENTITY: %s ----------" % company)
        x_test = pickle.load(open('intermediate/' + company + '_x_test.pkl', "rb"))
        y_test = pickle.load(open('intermediate/' + company + '_y_testDS.pkl', 'rb'))

        X_new = vectorizer.transform(x_test)
        pred = clf.predict(X_new)

        correct_count = 0
        for i, p in enumerate(pred):
            if p == y_test[i]:
                correct_count += 1
        print("number in test: %d" % len(pred))
        print("DS Model, percent correct: %0.3f" % ((float(correct_count)) / len(pred)))
