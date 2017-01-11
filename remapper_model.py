"""
author: audreyc
Remapper based on Fang's reverse mapping concepts.
"""

import os
import pickle
import DS_type_mapping
import time_extractor
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from sklearn import metrics
import time
import re
import urllib.request
import json

company_dir = 'sampledata2/'
pause_points = False
num_slots = 10
extra_slots = 10

def clean_probs(probs_list):
    """ returns a probability list without the really small values and sorted """
    sorted_list = []
    for prob_dict in probs_list:
        new_dict = {k: v for k, v in prob_dict.items() if v > 0.05}
        sorted_list.append(sorted(new_dict, key=new_dict.get, reverse=True))
    return sorted_list


def cross_check(ct, history_counter, allowed_list, max_pts=5):
    # print(history_counter.most_common(1)[0])
    if history_counter.most_common(1) and len(history_counter.most_common(1)[0][0]) > 5:
        ordered_names_all = [x[0].split('|')[0] for x in history_counter.most_common()]
    else:
        ordered_names_all = [x[0] for x in history_counter.most_common()]
    ordered_names = [y for y in ordered_names_all if y in allowed_list.keys()]
    if max_pts >= 10:
        multiplier = 2
        max_pts = int(max_pts / 2)
    else:
        multiplier = 1
    if pause_points:
        print(ordered_names)
    for e_num, et_key in enumerate(ordered_names):
        if e_num >= max_pts:
            break
        ct[et_key] += (max_pts - e_num) * multiplier
    # for a_num, a_key in enumerate(ordered_names_all):
    #     if a_num >= max_pts:
    #         continue
    #     ct[a_key] += max_pts - a_num
    return ct


def make_vector(existing_list, sector, list_labels, history_counter):
    top_ten = history_counter.most_common(num_slots)
    if len(top_ten) and len(top_ten[0][0]) > 5:
        # this catches automatically the case where the keys are '00210|blahblah'
        top_ten_labels = [v[0].split('|')[0] for v in top_ten]
    else:
        # most things have just five-digit keys
        top_ten_labels = [g[0] for g in top_ten]
    top_ten_values = [k[1] for k in top_ten]
    total_counts = sum(top_ten_values)
    for c in range(num_slots):
        if len(top_ten) > c and list_labels[c] in top_ten_labels:
            this_count = top_ten_values[top_ten_labels.index(list_labels[c])]
            existing_list[sector * num_slots + c] = (float(this_count) / float(total_counts))
    # print("vector list 0: ", existing_list[:39])
    # print("vector list 1: ", existing_list[40:79])
    # print("vector list 2: ", existing_list[80:119])
    # print("vector list 3: ", existing_list[120:159])
    # print("vector list 4: ", existing_list[160:])
    # input("pause")
    if sector != 2:
        return existing_list, list_labels
    # This should be the index of the end of the regular list
    # plus 5 slots each for leftover
    this_section = 4 * num_slots
    slot_counter = 0
    for index_history, d in enumerate(top_ten):
        if top_ten_labels[index_history] not in list_labels:
            existing_list[this_section + slot_counter] = float(d[1]) / total_counts
            list_labels.append(top_ten_labels[index_history])
            if pause_points:
                print(d[0])
            slot_counter += 1
            if slot_counter > extra_slots - 1:
                break
    return existing_list, list_labels


def run_ds_model(entity):
    print("---- Running DS model %s -----" % entity)
    start = time.time()

    x_train = pickle.load(open(company_dir + entity + "_X_train.pkl", "rb"))
    y_train = pickle.load(open(company_dir + entity + "_Y_trainDS.pkl", "rb"))
    x_test = pickle.load(open(company_dir + entity + "_X_test.pkl", "rb"))

    vectorizer = CountVectorizer(max_features=60000)
    x = vectorizer.fit_transform(x_train)
    print("Training set, num features: " + str(x.shape))

    clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
    clf.fit(x, y_train)
    pickle.dump(clf, open(company_dir + entity + '_DSmodel.pkl', 'wb'))
    pickle.dump(vectorizer.vocabulary_, open(company_dir + entity + '_DSvocab.pkl', 'wb'))

    x_new = vectorizer.transform(x_test)
    pred = clf.predict(x_new)
    pred_probs = clf.predict_proba(x_new)

    end = time.time()
    print("Time: " + str(end - start))
    pickle.dump(pred, open(company_dir + entity + '_DSpreds.pkl', 'wb'))


def run_ds_model_new(entity):
    print("---- Running DS model %s -----" % entity)
    start = time.time()

    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
    clf = joblib.load('new_ds_model/model.pkl')

    df = pd.read_csv('new_ds_model/expenseit_token_benchmark_dataset.tsv', sep='\t')

    x = vectorizer.transform(df.ocr_text)
    pred = clf.predict(x)

    end = time.time()
    print("Time: " + str(end - start))

    pickle.dump(pred, open(company_dir + entity + '_DSpreds.pkl', 'wb'))


def remap(entity, preds):
    test_info = pickle.load(open(company_dir + entity + "_info_test.pkl", "rb"))
    user_hist = pickle.load(open('new_ds_model/all_userhist.pkl', 'rb'))
    comp_hist = pickle.load(open('new_ds_model/all_companyhist.pkl', 'rb'))
    vend_hist = pickle.load(open('new_ds_model/all_vendorhist.pkl', 'rb'))
    amt_hist = pickle.load(open('inputdata/all_amounthist.pkl', 'rb'))
    allowed_types = pickle.load(open(company_dir + 'sampled_allowedtypes.pkl', 'rb'))
    real_answer = pickle.load(open(company_dir + entity + '_Y_testET.pkl', 'rb'))
    type_order, type_dict = DS_type_mapping.query_file()
    x_test = pickle.load(open(company_dir + entity + "_X_test.pkl", "rb"))
    clf = pickle.load(open('stackedmodel/clf.pkl', 'rb'))

    total_counter = Counter()
    correct = Counter()
    ds_correct = Counter()
    real_key_na = 0
    company_skipped = 0
    undef = 0
    used_extra_slots = 0
    no_entity_history = 0
    y_true = []
    y_pred = []
    for pred_index, pred in enumerate(preds):
        # if pause_points and pred != 'AIRFR':
        #     continue
        company = test_info[pred_index]['entity']
        userid = company + '-' + (test_info[pred_index]['userid'])
        vendor_raw = test_info[pred_index]['vendor']
        if vendor_raw:
            vendor_txt = re.sub("([^\w]|[ 0-9_])", '', vendor_raw.lower())
            # vendor = str(userid) + '-' + vendor_txt
            vendor = company + '-' + vendor_txt
        else:
            vendor = ''
        amount = test_info[pred_index]['amount']
        real_key = real_answer[pred_index]

        if company not in allowed_types.keys():
            company_skipped += 1
            continue
        else:
            allowed_list = allowed_types[company]
            if real_key not in allowed_list.keys():
                real_key_na += 1
                if pause_points:
                    print("real key not in allowed list")
                    print(real_key, allowed_list)
                continue

        real_ds_type = DS_type_mapping.to_ds_types(allowed_list[real_key], type_order, type_dict)

        new_allowed_list = {}
        for k, v in allowed_list.items():
            if v.startswith('z') or v.startswith('xx') or v.startswith('Z'):
                continue
            if DS_type_mapping.to_ds_types(v, type_order, type_dict) == pred:
                new_allowed_list[k] = v
        # if pause_points:
        #     print("Allowed Expense Types:", new_allowed_list)

        v_list = [0 for _ in range(4 * num_slots + extra_slots)]
        if company in comp_hist.keys():
            this_comp_hist = comp_hist[company].most_common(num_slots)
            list_labels = [x[0] for x in this_comp_hist if x[0] in new_allowed_list.keys()]
            list_labels.extend([0 for _ in range(num_slots - len(list_labels))])
            v_list, list_labels = make_vector(v_list, 0, list_labels, comp_hist[company])
        else:
            # make_vector(v_list, l_list, labels, Counter())
            no_entity_history += 1
            continue
        if pause_points:
            print(list_labels)
        total_counter[pred] += 1
        this_user_hist = user_hist[userid] if userid in user_hist.keys() else Counter()
        v_list, list_labels = make_vector(v_list, 1, list_labels, this_user_hist)
        this_vend_hist = vend_hist[vendor] if vendor in vend_hist.keys() else Counter()
        v_list, list_labels = make_vector(v_list, 2, list_labels, this_vend_hist)
        # This is the section where you give points (or subtract points) based on amount
        if amount and company in amt_hist.keys():
            amount = float(amount)
            if pause_points:
                print("Amount: ", str(amount), end=' ')
            if amount == 0:
                this_amnt_hist = amt_hist[company]['Zero']
            elif amount < 10:
                this_amnt_hist = amt_hist[company]['Under Ten']
            elif amount < 50:
                this_amnt_hist = amt_hist[company]['Under Fifty']
            elif amount < 100:
                this_amnt_hist = amt_hist[company]['Under Hundred']
            elif amount < 1000:
                this_amnt_hist = amt_hist[company]['Under Thousand']
            else:
                this_amnt_hist = amt_hist[company]['Other']
        else:
            this_amnt_hist = Counter()
        v_list, list_labels = make_vector(v_list, 3, list_labels, this_amnt_hist)

        if pause_points:
            print(v_list)
        et_pred_index = clf.predict([v_list])
        if et_pred_index < num_slots:
            et_pred = list_labels[et_pred_index]
        elif et_pred_index < len(list_labels):
            et_pred = list_labels[et_pred_index]
            used_extra_slots += 1
        else:
            print("out of bounds", len(list_labels), et_pred_index)
            # print(len(list_labels), et_pred_index)
        if et_pred == real_key:
            correct[pred] += 1

        if real_ds_type == pred:
            ds_correct[pred] += 1

        y_true.append(real_ds_type)
        y_pred.append(pred)

        if pause_points:
            print('Pred-DS:', pred, ', Real-DS:', real_ds_type, ', Real ET:', real_key, allowed_list[real_key], ', Pred ET:', et_pred)
            input('pause')

    print("ouput_name skipped: ", company_skipped)
    print("no entity history: ", no_entity_history)
    print("Real key not in allowed list: ", real_key_na, "(probably the entity changed their allowed types recently)")
    print("Total counter: ", sum(total_counter.values()), total_counter)
    print("DS Corrects: ", sum(ds_correct.values()), ds_correct)
    print("CORRECT ET GUESS: ", sum(correct.values()), correct)
    print("# undefs:", undef)
    print("used extra slots: ", used_extra_slots)

if __name__ == '__main__':
    company = 'sampled'
    if not os.path.isfile(company_dir + company + '_DSpreds.pkl'):
        run_ds_model(company)

    pred_list = pickle.load(open(company_dir + company + '_DSpreds.pkl', 'rb'))
    remap(company, pred_list)

