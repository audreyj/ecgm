import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
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
    amount_history = pickle.load(open("inputdata/" + company + "_typeamount.pkl", 'rb'))

    new_history = {}
    for k, v in amount_history.items():
        new_history[k.split('|')[0]] = v

    max_amt = 0
    max_class = pred[0]
    for typekey in classes:
        if typekey not in new_history.keys():
            continue
        tup = new_history[typekey]
        if tup[2] < 10:
            continue
        new_amt = tup[0] + tup[1]
        if new_amt > max_amt:
            max_class = typekey
            max_amt = new_amt

    new_pred = []
    for pred_index, prob_list in enumerate(pred_probs):
        amount = float(test_info[pred_index]['amount'])
        new_pred.append(adjust_predictions(pred[pred_index], amount, prob_list, classes, new_history, max_class))

    return new_pred


def select_dstypes(x_set, select_set):
    new_x = []
    for i, dstype in enumerate(select_set):
        if dstype == 'MEALS':
            if type(x_set) is list:
                new_x.append(x_set[i])
            else:
                new_x.append(i)
    if type(x_set) is list:
        output = new_x
    else:
        output = x_set[new_x]
    return output


start = time.time()

company = 'voith'
train_replace = ''
test_replace = ''
tr = train_replace if train_replace else 'train'
te = test_replace+'test' if test_replace else 'test'

X_train = pickle.load(open('intermediate/' + company + '_x_' + tr + 'ET.pkl', 'rb'))
y_train = pickle.load(open('intermediate/' + company + '_y_' + tr + 'ET.pkl', 'rb'))
train_select = pickle.load(open('intermediate/' + company + '_y_' + tr + 'DS.pkl', 'rb'))
X_test = pickle.load(open('intermediate/' + company + '_x_' + te + 'ET.pkl', "rb"))
y_test = pickle.load(open('intermediate/' + company + '_y_' + te + 'ET.pkl', "rb"))
test_select = pickle.load(open('intermediate/' + company + '_y_' + te + 'DS.pkl', 'rb'))

train_info = pickle.load(open('intermediate/' + company + '_info_' + tr + 'ET.pkl', 'rb'))
test_info = pickle.load(open('intermediate/' + company + '_info_' + te + 'ET.pkl', 'rb'))
# comp_hist = pickle.load(open('inputdata/all_companyhistET.pkl', 'rb'))

X_train_subset = select_dstypes(X_train, train_select)
y_train_subset = select_dstypes(y_train, train_select)
train_info_subset = select_dstypes(train_info, train_select)
test_info_subset = select_dstypes(test_info, test_select)
X_test_subset = select_dstypes(X_test, test_select)
y_test_subset = select_dstypes(y_test, test_select)
X_train = X_train_subset
y_train = y_train_subset
train_info = train_info_subset
test_info = test_info_subset
X_test = X_test_subset
y_test = y_test_subset

vectorizer = CountVectorizer(max_features=1200)
X = vectorizer.fit_transform(X_train)
X_new = vectorizer.transform(X_test)

X_dense = X.toarray()
amt_array = np.array([[float(x['amount'])] for x in train_info])
print(X_dense.shape, amt_array.shape)
X = np.hstack((X_dense, amt_array))

X_test_dense = X_new.toarray()
test_amt_list = np.array([[float(y['amount'])] for y in test_info])
X_new = np.hstack((X_test_dense, test_amt_list))

# X = np.array([[float(x['amount'])] for x in train_info_subset])
# X_new = np.array([[float(x['amount'])] for x in test_info_subset])

print("Training set, num features: " + str(X.shape))

clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
# clf = tree.DecisionTreeClassifier(random_state=1, class_weight='balanced')
# clf = ensemble.RandomForestClassifier(random_state=1)
clf.fit(X, y_train)
pickle.dump(clf, open('meals_amount_clf.pkl', 'wb'))

pred = clf.predict(X_new)
pred_probs = clf.predict_proba(X_new)
classes = clf.classes_

# Rejector
meals_pred = rejection(company, pred, pred_probs, classes, test_info)

print(metrics.classification_report(y_test, pred))
# big_mat = metrics.confusion_matrix(y_test_subset, pred)
# print(big_mat)
print("F1 Macro Score: %0.3f" % (metrics.f1_score(y_test, pred, average='macro')))
# print(X[-1], pred[-1], y_test[-1])
# print(user_hist[train_info[-1]['entity']+'-'+train_info[-1]['userid']])

correct_count = 0
in_meals = 0
for i, p in enumerate(pred):
    # print(y_test[i] + ', ' + str(test_info[i]['amount']), ' || ', p)
    if p == y_test[i]:
        correct_count += 1
    if y_test[i] in ['00090', '01150', '01181', '01188', '01189', '01197', 'BRKFT', 'DINNR', 'LUNCH']:
        in_meals += 1

print("percent correct: %0.3f" % ((float(correct_count))/len(pred)))
print("percent real answers in MEALS: %0.3f" % ((float(in_meals))/len(pred)))

end = time.time()
print("Time: " + str(end-start))
