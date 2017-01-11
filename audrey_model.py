"""
author: audreyc
Last Update: 04/27/16
Trains model for all companies and outputs a classifier in a pickle for future use
"""

import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
import DS_type_mapping as ds
import time


def adjust_predictions(current_pred, amount, probs, classes, history, max_class):
    if amount == 0:
        return current_pred
    prob_dict = {}
    for i, c in enumerate(classes):
        if probs[i] > 0.1:
            prob_dict[c] = probs[i]
    prob_list = sorted(prob_dict, key=prob_dict.get, reverse=True)
    for each_pred in prob_list:
        c_avg = history[each_pred][0]
        c_std = history[each_pred][1] * 2
        if (c_avg - c_std) < amount < (c_avg + c_std):
            # print("changed %s to %s (%f)" % (current_pred, each_pred, amount))
            return each_pred
    if amount > 100:
        return max_class
    return current_pred


def rejection(company, pred, pred_probs, classes, test_info):
    file_amount_history = pickle.load(open("inputdata/" + company + "_typeamount.pkl", 'rb'))
    to, td = ds.query_file()

    amount_history = {}
    for k, v in file_amount_history.items():
        if ds.to_ds_types(k.split('|')[1], to, td):
            amount_history[k.split('|')[0]] = v

    max_amt = 0
    total_history = 0
    max_class = pred[0]
    for typekey in classes:
        if typekey not in amount_history.keys():
            continue
        tup = amount_history[typekey]
        total_history += tup[2]
        if tup[2] < 10:
            continue
        new_amt = tup[0] + tup[1] * 2
        if new_amt > max_amt:
            max_class = typekey
            max_amt = new_amt

    new_pred = []
    for pred_index, prob_list in enumerate(pred_probs):
        amount = float(test_info[pred_index]['amount'])
        new_prob_list = []
        for prob_index, prob in enumerate(prob_list):
            if classes[prob_index] not in amount_history.keys():
                new_prob_list.append(prob)
            else:
                new_prob_list.append(prob + (amount_history[classes[prob_index]][2] / float(total_history)))
        new_pred.append(adjust_predictions(pred[pred_index], amount, new_prob_list, classes, amount_history, max_class))

    return new_pred


def vec_amount(info_list):
    # 0, 1-5, 5-10, 10-20, 20-50, 50-100, 100+
    return_array = np.zeros((len(info_list), 7), dtype=np.int)
    for i, x in enumerate(info_list):
        amount = float(x['amount'])
        if amount == 0:
            return_array[i][0] = 1
        elif amount < 5:
            return_array[i][1] = 1
        elif amount < 10:
            return_array[i][2] = 1
        elif amount < 20:
            return_array[i][3] = 1
        elif amount < 50:
            return_array[i][4] = 1
        elif amount < 100:
            return_array[i][5] = 1
        else:
            return_array[i][6] = 1
    return return_array


def select_dstypes(x_set, select_set):
    meals = []
    non_meals = []
    for i, dstype in enumerate(select_set):
        if dstype == 'MEALS':
            meals.append(x_set[i])
        else:
            non_meals.append(x_set[i])
    return meals, non_meals


def run_all(company):
    start = time.time()

    x_train = pickle.load(open('intermediate/' + company + '_x_train.pkl', 'rb'))
    y_train = pickle.load(open('intermediate/' + company + '_y_trainET.pkl', 'rb'))
    train_select = pickle.load(open('intermediate/' + company + '_y_trainDS.pkl', 'rb'))
    x_test = pickle.load(open('intermediate/' + company + '_x_test.pkl', "rb"))
    y_test = pickle.load(open('intermediate/' + company + '_y_testET.pkl', "rb"))
    test_select = pickle.load(open('intermediate/' + company + '_y_testDS.pkl', 'rb'))

    train_info = pickle.load(open('intermediate/' + company + '_info_train.pkl', 'rb'))
    test_info = pickle.load(open('intermediate/' + company + '_info_test.pkl', 'rb'))
    # comp_hist = pickle.load(open('inputdata/all_companyhistET.pkl', 'rb'))

    x_train_meals, x_train_nonmeals = select_dstypes(x_train, train_select)
    y_train_meals, y_train_nonmeals = select_dstypes(y_train, train_select)
    train_info_meals, train_info_nonmeals = select_dstypes(train_info, train_select)
    test_info_meals, test_info_nonmeals = select_dstypes(test_info, test_select)
    x_test_meals, x_test_nonmeals = select_dstypes(x_test, test_select)
    y_test_meals, y_test_nonmeals = select_dstypes(y_test, test_select)

    print(len(train_info_meals), train_info_meals[1])

    # Non-meals classification section
    print('NONMEALS Train set: %d, Test set: %d' % (len(x_train_nonmeals), len(x_test_nonmeals)))
    if len(x_train_nonmeals) and len(x_test_nonmeals):
        vec_nonmeals = CountVectorizer(max_features=60000)
        x_vec_train_nonmeals = vec_nonmeals.fit_transform(x_train_nonmeals)
        x_vec_test_nonmeals = vec_nonmeals.transform(x_test_nonmeals)
        print("Training set, num features: " + str(x_vec_train_nonmeals.shape))

        clf_nonmeals = linear_model.LogisticRegression(random_state=1, solver='liblinear')
        clf_nonmeals.fit(x_vec_train_nonmeals, y_train_nonmeals)
        pickle.dump(clf_nonmeals, open('intermediate/' + company + '_ACmodel_nonmeals.pkl', 'wb'))

        nonmeals_pred = clf_nonmeals.predict(x_vec_test_nonmeals)
        nonmeals_correct_count = 0
        for i, p in enumerate(nonmeals_pred):
            # print(y_test[i] + ', ' + str(test_info[i]['amount']), ' || ', p)
            if p == y_test_nonmeals[i]:
                nonmeals_correct_count += 1
        print("NONMEALS accuracy: %0.3f" %
              ((float(nonmeals_correct_count)) / len(nonmeals_pred)))

    # Meals classification section
    print('MEALS Train set: %d, Test set: %d' % (len(x_train_meals), len(x_test_meals)))
    if len(x_train_meals) and len(x_test_meals):
        vec_meals = CountVectorizer(max_features=1200)
        x_vec_train_meals = vec_meals.fit_transform(x_train_meals)
        x_vec_test_meals = vec_meals.fit_transform(x_test_meals)
        x_dense = x_vec_train_meals.toarray()
        amt_array = vec_amount(train_info_meals)
        # amt_array = np.array([[float(x['amount'])] for x in train_info_meals])
        # print(x_dense.shape, amt_array.shape)
        x_vec_train_meals = np.hstack((x_dense, amt_array))
        x_test_dense = x_vec_test_meals.toarray()
        test_amt_array = vec_amount(test_info_meals)
        # test_amt_array = np.array([[float(y['amount'])] for y in test_info_meals])
        x_vec_test_meals = np.hstack((x_test_dense, test_amt_array))
        print("Training set, num features: " + str(x_vec_train_meals.shape))

        clf_meals = linear_model.LogisticRegression(random_state=1, solver='liblinear')
        clf_meals.fit(x_vec_train_meals, y_train_meals)
        pickle.dump(clf_meals, open('intermediate/' + company + '_ACmodel_meals.pkl', 'wb'))

        meals_pred = clf_meals.predict(x_vec_test_meals)
        meals_pred_probs = clf_meals.predict_proba(x_vec_test_meals)
        meals_classes = clf_meals.classes_

        # Rejector
        meals_pred = rejection(company, meals_pred, meals_pred_probs, meals_classes, test_info_meals)

        meals_correct_count = 0
        for i, p in enumerate(meals_pred):
            # print(y_test[i] + ', ' + str(test_info[i]['amount']), ' || ', p)
            if p == y_test_meals[i]:
                meals_correct_count += 1
        print("MEALS accuracy: %0.3f" % ((float(meals_correct_count)) / len(meals_pred)))

    end = time.time()
    print("Time: " + str(end-start))
