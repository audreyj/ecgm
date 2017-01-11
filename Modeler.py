import pickle
import numpy as np
import re
import fuzzywuzzy
from fuzzywuzzy import process
import DS_type_mapping
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
import time


def use_history(probability_list, mapper, history_dict):
    ordered_history = [v[0].split('|')[1] for v in history_dict.most_common()]
    output = Counter()
    for p in probability_list:
        h = process.extract(mapper[p], ordered_history)
        for result in h:
            if result[0] in output:
                output[result[0]] += result[1]
            else:
                output[result[0]] = result[1]
    return output


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

X_train = pickle.load(open("intermediate/concur_x_trainETN.pkl", 'rb'))
y_train = pickle.load(open("intermediate/concur_y_trainETN.pkl", 'rb'))
train_select = pickle.load(open('intermediate/concur_y_trainDS.pkl', 'rb'))
X_test = pickle.load(open("intermediate/concur_x_jantestETN.pkl", "rb"))
y_test = pickle.load(open("intermediate/concur_y_jantestETN.pkl", "rb"))
test_select = pickle.load(open('intermediate/concur_y_jantestDS.pkl', 'rb'))

X_train_subset = select_dstypes(X_train, train_select)
y_train_subset = select_dstypes(y_train, train_select)
X_test_subset = select_dstypes(X_test, test_select)
y_test_subset = select_dstypes(y_test, test_select)
X_train = X_train_subset
y_train = y_train_subset
X_test = X_test_subset
y_test = y_test_subset

vectorizer = CountVectorizer(max_features=1200)
X = vectorizer.fit_transform(X_train)
print("Training set, num features: " + str(X.shape))

# clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
clf = ensemble.RandomForestClassifier(random_state=1)
secondvec = CountVectorizer()
Y_temp = secondvec.fit_transform(y_train)
Y = Y_temp.toarray()
y_out_temp = secondvec.transform(y_test)
y_out = y_out_temp.toarray()
clf.fit(X, Y)

X_new = vectorizer.transform(X_test)
pred = clf.predict(X_new)
pred_probs = clf.predict_proba(X_new)
classes = clf.classes_
vocab = secondvec.get_feature_names()
print(vocab)

# new_probs = [{} for y in old_pred]
# for voc_index, voc in enumerate(pred_probs):
#     for pred_index, each_pred in enumerate(voc):
#         if each_pred[1] > 0.1:
#             new_probs[pred_index].update({voc_index: each_pred[1]})
# pred_probs = []  # wipe pred_probs
# for prob_list in new_probs:
#     pred_probs.append(sorted(prob_list, key=prob_list.get, reverse=True))

# test_info = pickle.load(open("intermediate/info_janET.pkl", 'rb'))
# user_hist = pickle.load(open('inputdata/concur_userhistET.pkl', 'rb'))
# comp_hist = pickle.load(open('inputdata/concur_companyhistET.pkl', 'rb'))
# pred = []
# count_inhistory = 0
# for i, p in enumerate(pred_probs):
#     userid = test_info[i]['userid']
#     entity = test_info[i]['entity']
#     user_key = entity + '-' + userid
#     # if user_key in user_hist.keys():
#     #     pred.append(use_history(p, vocab, user_hist[user_key]))
#     #     # if y_test[i] in [v[0].split('|')[1] for v in user_hist[user_key].most_common()]:
#     #     #     count_inhistory += 1
#     if entity in comp_hist.keys():
#         pred.append(use_history(p, vocab, comp_hist[entity]))
#         # if y_test[i] in [v[0].split('|')[1] for v in comp_hist[entity].most_common()]:
#         #     count_inhistory += 1
#         # else:
#         #     print(y_test[i])
#     else:
#         pred.append([vocab[m] for m in p])

# print("percent in user history: %0.3f" % ((float(count_inhistory))/len(pred_probs)))

# print(metrics.classification_report(y_out, pred))
# big_mat = metrics.confusion_matrix(y_test, pred, labels=tops)
# print(big_mat)
# print("F1 Macro Score (of top types): %0.3f" % (metrics.f1_score(y_test, pred, average='macro')))
# the_test = 1  # Test index to print
# print(len(new_probs), len(new_probs[the_test]))
# print(X_test[the_test], new_probs[the_test], y_test[the_test], list(np.nonzero(y_out[the_test])[0]))

correct_count = 0
probs_out = []
for pred_index, real_answer in enumerate(y_test):
    p_temp = list(np.nonzero(pred[pred_index])[0])
    prediction = [vocab[x] for x in p_temp]
    # print(real_answer, " || ", prediction)
    realsub = re.sub('[^0-9a-zA-Z]+', ' ', real_answer)
    reallist = realsub.lower().split()
    all_in = 1
    for el in reallist:
        if el not in prediction:
            all_in = 0
    correct_count = correct_count+1 if all_in else correct_count
    if not all_in:
        print(pred_index, real_answer, ' || ', prediction)
print("percent correct: %0.3f" % ((float(correct_count))/len(pred)))

# importances = clf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Top Ten Feature ranking:")
# for f in range(10):
#     print("%d. feature %s (%f)" % (f + 1, vectorizer.get_feature_names()[indices[f]], importances[indices[f]]))

# pickle.dump(pred_probs, open('intermediate/probs_train.pkl', 'wb'))
end = time.time()
print("Time: " + str(end-start))
