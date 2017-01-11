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


def check_type(probs, classes, allowed):
    max_probs = sorted(list(set(probs)), reverse=True)
    ordered_allowed = [v[0] for v in allowed.most_common()]
    # print("max probs: " + str(max_probs))
    for prob in max_probs:
        if prob == 0:
            return ordered_allowed[0]
        max_indices = np.argwhere(probs == prob).flatten().tolist()
        max_classes = [classes[x] for x in max_indices]
        # print("max indices: " + str(max_indices))
        for m in ordered_allowed:
            # print("class: " + str(classes[m]))
            if m in max_classes:
                return m

concur_history_total = [('00101|Taxi-Shuttle-Train', 9751), ('00090|Business Meals - Meetings', 9685),
                        ('DINNR|Individual Dinner', 6873), ('00094|Parking - Tolls', 5001),
                        ('LUNCH|Individual Lunch', 4757), ('BRKFT|Individual Breakfast', 3287),
                        ('01181|Beverages', 2371), ('CELPH|Cellular - Mobile Phone', 1368),
                        ('OFCSP|Office Supplies', 1320), ('GASXX|Gas - Petrol (rental car only)', 1126),
                        ('AIRFR|Airfare', 1075), ('ENTOT|Entertainment', 690), ('CARRT|Car Rental', 540),
                        ('00008|Internet Access', 529), ('00091|Company - Employee Events', 395),
                        ('01200|Marketing Events ', 343), ('GIFTS|Gifts - Incentives (Employee)', 293),
                        ('01151|Internet Access - Travel', 253), ('01144|Gifts (Non-employee)', 242),
                        ('00033|Other Travel Expenses', 210), ('01171|Airfare Fees', 194), ('00100|Equipment.', 150),
                        ('01150|Subsistence Meal (>1 Employee)', 149), ('01197|Beverages - Alcohol', 144),
                        ('00013|Miscellaneous Expense', 122), ('HOMPH|Local Phone', 115),
                        ('00092|Conf - Seminar - Trng', 111), ('00016|Software', 96),
                        ('01142|Entertainment - Staff', 84), ('01203|Parking Subsidy (Bellevue Employees Only)', 77),
                        ('00005|Books and Reference Material', 76), ('01201|Collateral/Sales Tools', 66),
                        ('00017|Shipping', 65), ('POSTG|Postage', 57), ('00110|Booking Fees', 55),
                        ('DUESX|Membership Dues', 45), ('01141|Entertainment - Other', 36),
                        ('SEMNR|Printing Expenses', 34), ('01188|Business Meals - Meetings In France', 33),
                        ('TRDSH|Trade Shows', 32), ('01196|Transit and Train', 30),
                        ('01189|Business Meals - Meetings Abroad', 26), ('TELPH|Long Distance', 23),
                        ('MILEG|Mileage (personal car only)', 23), ('01202|Expatriate Employee Expenses', 19),
                        ('01126|Sales Incentives', 19), ('01143|Home Business Line', 16),
                        ('00095|Publications - Subscriptions', 16), ('LNDRY|Laundry', 11), ('01130|Website Fees', 10),
                        ('LODNG|Hotel', 10), ('00031|Tips', 7), ('01216|Equipment', 4), ('UNDEF|Undefined', 3),
                        ('01194|Insurance', 2), ('01122|Promotional Materials', 1), ('FAXXX|Fax', 1),
                        ('01187|Hotel Employee Abroad', 1), ('00030|American Express Fees', 1)]


def add_features(original_list, info_list, ocr=0):
    # X = []
    # count_users = 0
    # count_comps = 0
    # for pred_index, prediction in enumerate(original_list):
    #     entity = info_list[pred_index]['entity']
    #     userid = info_list[pred_index]['userid']
    #     user_key = entity + '-' + userid
    #     X.append([])
    #     if user_key in user_hist.keys():
    #         count_users += 1
    #         max_num = max(user_hist[user_key].values())
    #         this_history = {x.split('|')[0]: y for x, y in user_hist[user_key].items()}
    #         # print(this_history)
    #     elif entity in comp_hist.keys():
    #         count_comps += 1
    #         max_num = max(comp_hist[entity].values())
    #         this_history = {x.split('|')[0]: y for x, y in comp_hist[entity].items()}
    #     else:
    #         max_num = 1
    #         this_history = {}
    #     for tup in concur_history_total:
    #         expense_key = tup[0].split('|')[0]
    #         if expense_key in this_history.keys():
    #             # print(max_num, expense_key)
    #             X[pred_index].append((float(this_history[expense_key]))/max_num)
    #         else:
    #             X[pred_index].append(0)
    # temp = np.array(X)
    # if not ocr:
    #     output_array = np.hstack((original_list, temp))
    #     print(output_array.shape, count_users, count_comps)
    # else:
    if ocr:
        # print(temp.shape)
        print(original_list.shape)
        vectorizer = CountVectorizer(max_features=1200)
        ocr_array = vectorizer.fit_transform(ocr)
        ocr_dense = ocr_array.toarray()
        print(ocr_dense.shape)
        output_array = np.hstack((ocr_dense, original_list))
    return output_array


start = time.time()

X_train = pickle.load(open('intermediate/probs_train.pkl', 'rb'))
y_train = pickle.load(open("intermediate/concur_y_trainET.pkl", 'rb'))
X_test = pickle.load(open("intermediate/probs_jan.pkl", "rb"))
y_test = pickle.load(open("intermediate/concur_y_jantestET.pkl", "rb"))
train_info = pickle.load(open("intermediate/testset_info.pkl", 'rb'))
test_info = pickle.load(open("intermediate/info_janET.pkl", 'rb'))
user_hist = pickle.load(open('inputdata/all_userhistET.pkl', 'rb'))
comp_hist = pickle.load(open('inputdata/all_companyhistET.pkl', 'rb'))
ocr_train = pickle.load(open('intermediate/concur_x_trainET.pkl', 'rb'))
ocr_test = pickle.load(open('intermediate/concur_x_jantestET.pkl', 'rb'))

# X = add_features(X_train, train_info, ocr_train)
# X_new = add_features(X_test, test_info, ocr_test)
X = X_train
X_new = X_test

print("Training set, num features: " + str(X.shape))

clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
# clf = tree.DecisionTreeClassifier(random_state=1, class_weight='balanced')
# clf = ensemble.RandomForestClassifier(random_state=1)
clf.fit(X, y_train)

# X_new = vectorizer.transform(X_test)
pred = clf.predict(X_new)
pred_probs = clf.predict_proba(X_new)
classes = clf.classes_

# big_mat = metrics.confusion_matrix(y_test, pred, labels=tops)
print(metrics.classification_report(y_test, pred))
# print(big_mat)
print("F1 Macro Score (of top types): %0.3f" % (metrics.f1_score(y_test, pred, average='macro')))
# print(X[-1], pred[-1], y_test[-1])
# print(user_hist[train_info[-1]['entity']+'-'+train_info[-1]['userid']])

correct_count = 0
for i, p in enumerate(pred):
    # print(y_test[i], " || ", p)
    if p == y_test[i]:
        correct_count += 1
print("percent correct: %0.3f" % ((float(correct_count))/len(pred)))

end = time.time()
print("Time: " + str(end-start))
