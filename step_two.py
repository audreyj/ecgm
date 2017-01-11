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

start = time.time()

X_train = pickle.load(open("intermediate/concur_x_train2ET.pkl", "rb"))
y_train = pickle.load(open("intermediate/concur_y_train2ET.pkl", "rb"))
# X_train = pickle.load(open("intermediate/janpred_probsDS.pkl", 'rb'))
# y_train = pickle.load(open("intermediate/concur_y_jantestET.pkl", 'rb'))
X_test = pickle.load(open("intermediate/concur_x_train2DS.pkl", "rb"))
y_test = pickle.load(open("intermediate/concur_y_train2DS.pkl", "rb"))
# X_test = pickle.load(open("intermediate/concur_x_jantestDS.pkl", "rb"))
# y_test = pickle.load(open("intermediate/concur_y_jantestDS.pkl", "rb"))

vectorizer = CountVectorizer(max_features=60000)
X = vectorizer.fit_transform(X_train)
print("Training set, num features: " + str(X.shape))

clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
# clf = ensemble.RandomForestClassifier(random_state=1)
clf.fit(X, y_train)

X_new = vectorizer.transform(X_test)
pred = clf.predict(X_new)
pred_probs = clf.predict_proba(X_new)
classes = clf.classes_

print(metrics.classification_report(y_test, pred))
print("F1 Macro Score (of top types): %0.3f" % (metrics.f1_score(y_test, pred, average='macro')))
print(X_test[-1], pred[-1], y_test[-1])

correct_count = 0
for i, p in enumerate(pred):
    if p == y_test[i]:
        correct_count += 1
print("percent correct: %0.3f" % ((float(correct_count))/len(pred)))

pickle.dump(pred_probs, open('intermediate/probs_train.pkl', 'wb'))
end = time.time()
print("Time: " + str(end-start))
