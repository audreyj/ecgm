"""
author: audreyc
Last Update: 04/28/16
This code is meant to run on the compute machine only due to size and time requirements
This code should run all steps in sequence for all companies with proper save out points
at each step.  Starting from inputdata/all_trainset
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

# company_list = ['p0000745jr8u', 'p0003884x7lt', 'p0043611aoji', 'p0039557fvbf',
#                 'p0090051h2oq', 'p0006679vz2s', 'p00425z4gu', 'p0009976ed7k',
#                 'p0005859huep', 'p0089280mxzg', 'p0079383unbs', 'p00521862bvn',
#                 'p0002406hlem', 'p0002233vdsm', 'p0096373arhm', 'p0039096va39',
#                 'p0046598vm29', 'p0036394dcvf', 'p0081759udz2', 'p0079434mdig',
#                 'p0014655asb9', 'p0043883wrzh']
company_list = ['p00425z4gu']
company_dir = 'intermediate/'
# company_list = pickle.load(open('inputdata/all_company_list.pkl', 'rb'))
# company_dir = 'companydata/'


def expense_type_model(entity):
    start = time.time()

    x_train = pickle.load(open(company_dir + entity + "_x_train.pkl", "rb"))
    y_train = pickle.load(open(company_dir + entity + "_y_trainET.pkl", "rb"))
    x_test = pickle.load(open(company_dir + entity + "_x_test.pkl", "rb"))
    y_test = pickle.load(open(company_dir + entity + "_y_testET.pkl", "rb"))

    # Adding logic for too few trainings or all the same class
    if len(set(y_train)) < 2:
        return len(y_test), len(y_test)
    if len(y_test) < 10:
        return 0, len(y_test)

    vectorizer = CountVectorizer(max_features=10000)
    x = vectorizer.fit_transform(x_train)
    # print("Training set, num features: " + str(x.shape))

    clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
    # clf = ensemble.RandomForestClassifier(random_state=1)
    clf.fit(x, y_train)
    pickle.dump(clf, open('intermediate/'+entity+'_ETmodel.pkl', 'wb'))
    pickle.dump(vectorizer.vocabulary_, open('intermediate/'+entity+'_vectorizer.pkl', 'wb'))

    x_new = vectorizer.transform(x_test)
    pred = clf.predict(x_new)
    # pred_probs = clf.predict_proba(x_new)
    # classes = clf.classes_

    # print(metrics.classification_report(y_test, pred))
    # print(x_test[-1], pred[-1], y_test[-1])

    correct_count = 0
    for i, p in enumerate(pred):
        if p == y_test[i]:
            correct_count += 1
    accuracy = (float(correct_count))/len(pred)
    print("number in test: %d" % len(pred))
    print("ET Model, percent correct: %0.3f" % accuracy)
    # print("%s, %d, %d, %d, %0.4f" % (entity, int(x.shape[0]), int(x.shape[1]), len(pred), accuracy))

    # pickle.dump(pred_probs, open('intermediate/probs_train.pkl', 'wb'))
    end = time.time()
    print("Time: " + str(end-start))

    return correct_count, len(pred)


if __name__ == '__main__':
    # if not os.path.isfile('inputdata/'+company_list[0]+'_trainset'):
    #     gen_files.gen_files()
    all_correct = 0
    all_total = 0.0
    # print("EntityID, Training set count, Training set num features, Test set count, ET model accuracy")
    for company in company_list:
        print("-------- ENTITY: %s ----------" % company)
        # step_zero.parse_file(ouput_name)
        company_correct, company_total = expense_type_model(company)
        all_correct += company_correct
        all_total += company_total
        # audrey_model.run_all(ouput_name)
    print("----------- FINAL (num companies = %d) -----------" % len(company_list))
    print("Total Accuracy = %0.3f" % (float(all_correct) / all_total))
