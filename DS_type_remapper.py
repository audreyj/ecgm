"""
author: audreyc
"""

import os
import pickle
import DS_type_mapping_v2
import time_extractor
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
import time
import re
import urllib.request
import json

company_dir = 'sampledata2/'
pause_points = False


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

    end = time.time()
    print("Time: " + str(end - start))

    pickle.dump(pred, open(company_dir + entity + '_DSpreds.pkl', 'wb'))


def run_ds_model_new(entity):
    print("---- Running DS model %s -----" % entity)
    start = time.time()

    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
    clf = joblib.load('new_ds_model/dstype_passive_aggressive_2.pkl')

    df = pd.read_csv('new_ds_model/expenseit_token_benchmark_dataset.tsv', sep='\t')

    x = vectorizer.transform(df.ocr_text)
    pred = clf.predict(x)

    end = time.time()
    print("Time: " + str(end - start))

    pickle.dump(pred, open(company_dir + entity + '_DSpreds.pkl', 'wb'))


def remap(file_name, preds):
    test_info = pickle.load(open(company_dir + file_name + "_info_test.pkl", "rb"))
    user_hist = pickle.load(open('new_ds_model/all_userhist.pkl', 'rb'))
    comp_hist = pickle.load(open('new_ds_model/all_companyhist.pkl', 'rb'))
    all_entity_hist = pickle.load(open('new_ds_model/allexpense_entityhistory.pkl', 'rb'))
    vend_hist = pickle.load(open('new_ds_model/all_vendorhist.pkl', 'rb'))
    uvend_hist = pickle.load(open('new_ds_model/all_uservendorhist.pkl', 'rb'))
    amt_hist = pickle.load(open('inputdata/all_amounthist.pkl', 'rb'))
    allowed_types = pickle.load(open(company_dir + 'sampled_allowedtypes.pkl', 'rb'))
    real_answer = pickle.load(open(company_dir + file_name + '_Y_testET.pkl', 'rb'))
    x_test = pickle.load(open(company_dir + file_name + "_X_test.pkl", "rb"))

    total_counter = Counter()
    correct = Counter()
    ds_correct = Counter()
    used_backup = Counter()
    real_key_na = 0
    user_history_exists = 0
    in_user_history = 0
    entity_history_exists = 0
    in_entity_history = 0
    vendor_history_exists = 0
    in_vendor_history = 0
    uvendor_history_exists = 0
    in_uvendor_history = 0
    company_skipped = 0
    undef = 0
    fangs_correct = Counter()

    for pred_index, pred in enumerate(preds):
        if pred_index > 1000:
            break
        # if pause_points and pred != 'AIRFR':
        #     continue
        company = test_info[pred_index]['entity']
        userid = company + '-' + (test_info[pred_index]['userid'])
        vendor_raw = test_info[pred_index]['vendor']
        if vendor_raw:
            vendor_txt = re.sub("([^\w]|[ 0-9_])", '', vendor_raw.lower())
            # vendor = str(userid) + '-' + vendor_txt
            vendor = company + '-' + vendor_txt
            uvendor = userid + '-' + vendor_txt
        else:
            vendor = ''
            uvendor = ''
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

        real_ds_type = DS_type_mapping_v2.to_ds_types(allowed_list[real_key])
        if pause_points:
            print(x_test[pred_index])
        new_allowed_list = {}
        for k, v in allowed_list.items():
            if v.startswith('z') or v.startswith('xx') or v.startswith('Z'):
                continue
            if DS_type_mapping_v2.to_ds_types(v) == pred:
                new_allowed_list[k] = v
        if pause_points:
            print("Allowed Expense Types:", new_allowed_list)

        et_guess = Counter()
        for et_key in new_allowed_list.keys():
            et_guess[et_key] += 50

        # This is the section that deals with time in Meals.
        if pred == "MEALS":
            te = time_extractor.TimeExtractor()
            minutes_from_midnight = te.extract_time(x_test[pred_index])
            if minutes_from_midnight['time'] and minutes_from_midnight['time'][0]['value'] != -1:
                mfm = minutes_from_midnight['time'][0]['value']
                if 240 < mfm < 630:  # 4:00am - 10:30am
                    searchstr = ['brk', 'bfast', 'break']
                elif 630 < mfm < 840:  # 2:00pm
                    searchstr = ['lun']
                elif mfm > 1020:  # 5:00pm
                    searchstr = ['din']
                else:
                    searchstr = []
                if pause_points:
                    print("Hours from Midnight: ", str(float(mfm) / 60), searchstr)
                for k, v in new_allowed_list.items():
                    for s in searchstr:
                        if s in v.lower():
                            et_guess[k] += 8
            # print(et_guess)
            # print(real_ds_type, real_key)

        # It's always XXXXX
        # if pred == 'LODNG':
        #     et_guess['LODNG'] += 3
        # if pred == 'CARRT':
        #     et_guess['CARRT'] += 3
        # if pred == 'TELEC':
        #     et_guess['CELPH'] += 1
        # if pred == 'OFCSP':
        #     et_guess['OFCSP'] += 3
        # if pred == 'GRTRN':
        #     et_guess['TAXIX'] += 3
        # if pred == 'PARKG':
        #     et_guess['PARKG'] += 3

        # This is the section where you give points (or subtract points) based on amount
        if amount and company in amt_hist.keys():
            amount = float(amount)
            if pause_points:
                print("Amount: ", str(amount), end=' ')
            if amount == 0:
                et_guess = cross_check(et_guess, amt_hist[company]['Zero'], new_allowed_list, 5)
            elif amount < 10:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Ten'], new_allowed_list, 10)
            elif amount < 50:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Fifty'], new_allowed_list, 10)
            elif amount < 100:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Hundred'], new_allowed_list, 10)
            elif amount < 1000:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Thousand'], new_allowed_list, 12)
            else:
                et_guess = cross_check(et_guess, amt_hist[company]['Other'], new_allowed_list, 14)

        total_counter[pred] += 1
        if vendor and vendor in vend_hist.keys():
            vendor_history_exists += 1
            if pause_points:
                print("Vendor Hist: ", vendor, end=' ')
            et_guess = cross_check(et_guess, vend_hist[vendor], new_allowed_list, 7)
            if real_key in vend_hist[vendor]:
                in_vendor_history += 1
        if uvendor and uvendor in uvend_hist.keys():
            uvendor_history_exists += 1
            if pause_points:
                print("User Vendor History: ", uvendor, end=' ')
            et_guess = cross_check(et_guess, uvend_hist[uvendor], new_allowed_list, 7)
            if real_key in uvend_hist[uvendor]:
                in_uvendor_history += 1
        if userid in user_hist.keys():
            user_history_exists += 1
            if pause_points:
                print("User Hist: ", userid, end=' ')
            et_guess = cross_check(et_guess, user_hist[userid], new_allowed_list, 6)
            if real_key in user_hist[userid]:
                in_user_history += 1
        if company in comp_hist.keys():
            entity_history_exists += 1
            if pause_points:
                print("Entity Hist: ", company, end=' ')
            et_guess = cross_check(et_guess, comp_hist[company], new_allowed_list, 5)
            if real_key in comp_hist[company]:
                in_entity_history += 1

        if not len(et_guess):
            used_backup[pred] += 1
            if pause_points:
                print(x_test[pred_index])
            if vendor and vendor in vend_hist.keys():
                et_guess = cross_check(et_guess, vend_hist[vendor], allowed_list, 5)
            elif userid in user_hist.keys():
                et_guess = cross_check(et_guess, user_hist[userid], allowed_list, 5)
            elif company in comp_hist.keys():
                et_guess = cross_check(et_guess, comp_hist[company], allowed_list, 5)
            elif company in all_entity_hist.keys():
                et_guess = cross_check(et_guess, all_entity_hist[company], new_allowed_list, 5)
            travel_words = ['trav', 't&', '&e', 'trans']
            for expkey, expname in allowed_list.items():
                if any([t in expname.lower() for t in travel_words]):
                    et_guess[expkey] += 1
            hotel_words = ['room', 'resort', 'lodging', 'hotel']
            if any([f in x_test[pred_index] for f in hotel_words]):
                for expkey, expname in allowed_list.items():
                    if any([g in expname.lower() for g in hotel_words]):
                        et_guess[expkey] += 1
        if not len(et_guess):
            undef += 1
            et_guess['UNDEF'] += 1
            print('allowed list: ', allowed_list)
            print(x_test[pred_index])
            print('ds pred: ', pred)
            print('real key: ', real_key)
            print('----------------------------')

        if pause_points:
            print("Final Guess rankings: ", et_guess)
        if et_guess.most_common(1)[0][0] == real_key:
            correct[real_ds_type] += 1

        if real_ds_type == pred:
            ds_correct[pred] += 1

        if 0:
            # Call API running in prod
            ds_api_url = 'http://seapr1dsweb.concurasp.com:80/ds-webapi/service/expenseClassification/receiptTypeClassification'
            request_type = {'Content-Type': 'application/json'}
            input_data = {'entityId': company, 'userId': test_info[pred_index]['userid'],
                          'ocrText': x_test[pred_index], 'cteLoginId': 'blank',
                          'userExpenseTypes': [{'ExpKey': 'AIRFR', 'Name': 'airfare'}]}
            input_data = json.dumps(input_data).encode('utf-8', 'ignore')
            req = urllib.request.Request(ds_api_url, input_data, request_type)
            f = urllib.request.urlopen(req)
            response_raw = f.read().decode('utf-8')
            response = json.loads(response_raw)
            fangs_et = response['expenseTypes'][0]['type'] if response['expenseTypes'] else ''
            # fangs_en = test_info[pred_index]['expense_type_name']
            # fangs_ds_type = DS_type_mapping.to_ds_types(fangs_en, type_order, type_dict)
            if fangs_et == real_key:
                fangs_correct[real_ds_type] += 1

        if pause_points:
            print('Pred-DS:', pred, ', Real-DS:', real_ds_type, ', Real ET:', real_key, allowed_list[real_key])
            input('pause')

    print("ouput_name skipped: ", company_skipped)
    print("Real key not in allowed list: ", real_key_na, "(probably the file_name changed their allowed types recently)")
    print("Total counter: ", sum(total_counter.values()), total_counter)
    print("DS Corrects: ", sum(ds_correct.values()), ds_correct)
    print("CORRECT ET GUESS: ", sum(correct.values()), correct)
    print("USED BACKUP: ", sum(used_backup.values()), used_backup)
    print("FANG'S CORRECT: ", sum(fangs_correct.values()), fangs_correct)
    print("# correct key somewhere in user hist / user history exists:",
          in_user_history, user_history_exists)
    print("# correct key somewhere in file_name hist / file_name history exists:",
          in_entity_history, entity_history_exists)
    print("# correct key somewhere in vendor hist / vendor history exists:",
          in_vendor_history, vendor_history_exists)
    print("# correct key somewhere in user-vendor hist / user-vendor history exists:",
          in_uvendor_history, uvendor_history_exists)
    print("# undefs:", undef)

if __name__ == '__main__':
    ouput_name = 'sampled'
    if not os.path.isfile(company_dir + ouput_name + '_DSpreds.pkl'):
        run_ds_model_new(ouput_name)

    pred_list = pickle.load(open(company_dir + ouput_name + '_DSpreds.pkl', 'rb'))
    remap(ouput_name, pred_list)

