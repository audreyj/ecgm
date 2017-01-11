"""
author: audreyc
"""

import os
import pickle
import time_extractor
from collections import Counter
import time
import re

company_dir = 'sampledata2/'
pause_points = False


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


def remap():
    test_info = pickle.load(open(company_dir + "sampled_info_test.pkl", "rb"))
    user_hist = pickle.load(open('new_ds_model/all_userhist.pkl', 'rb'))
    comp_hist = pickle.load(open('new_ds_model/all_companyhist.pkl', 'rb'))
    all_entity_hist = pickle.load(open('new_ds_model/allexpense_entityhistory.pkl', 'rb'))
    vend_hist = pickle.load(open('new_ds_model/all_vendorhist.pkl', 'rb'))
    uvend_hist = pickle.load(open('new_ds_model/all_uservendorhist.pkl', 'rb'))
    amt_hist = pickle.load(open('inputdata/all_amounthist.pkl', 'rb'))
    allowed_types = pickle.load(open(company_dir + 'sampled_allowedtypes.pkl', 'rb'))
    real_answer = pickle.load(open(company_dir + 'sampled_Y_testET.pkl', 'rb'))
    x_test = pickle.load(open(company_dir + "sampled_X_test.pkl", "rb"))

    total_counter = 0
    correct = 0
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
    for pred_index, ocr_text in enumerate(x_test):
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

        et_guess = Counter()
        for et_key in allowed_list.keys():
            et_guess[et_key] += 1

        # This is the section that deals with time in Meals.
        # te = time_extractor.TimeExtractor()
        # minutes_from_midnight = te.extract_time(ocr_text)
        # if minutes_from_midnight['time'] and minutes_from_midnight['time'][0]['value'] != -1:
        #     mfm = minutes_from_midnight['time'][0]['value']
        #     if 240 < mfm < 630:  # 4:00am - 10:30am
        #         searchstr = ['brk', 'bfast', 'break']
        #     elif 630 < mfm < 840:  # 2:00pm
        #         searchstr = ['lun']
        #     elif mfm > 1020:  # 5:00pm
        #         searchstr = ['din']
        #     else:
        #         searchstr = []
        #     if pause_points:
        #         print("Hours from Midnight: ", str(float(mfm) / 60), searchstr)
        #     for k, v in allowed_list.items():
        #         for s in searchstr:
        #             if s in v.lower():
        #                 et_guess[k] += 8

        # This is the section where you give points (or subtract points) based on amount
        if amount and company in amt_hist.keys():
            amount = float(amount)
            if pause_points:
                print("Amount: ", str(amount), end=' ')
            if amount == 0:
                et_guess = cross_check(et_guess, amt_hist[company]['Zero'], allowed_list, 5)
            elif amount < 10:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Ten'], allowed_list, 10)
            elif amount < 50:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Fifty'], allowed_list, 10)
            elif amount < 100:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Hundred'], allowed_list, 10)
            elif amount < 1000:
                et_guess = cross_check(et_guess, amt_hist[company]['Under Thousand'], allowed_list, 12)
            else:
                et_guess = cross_check(et_guess, amt_hist[company]['Other'], allowed_list, 14)

        total_counter += 1
        if vendor and vendor in vend_hist.keys():
            vendor_history_exists += 1
            if pause_points:
                print("Vendor Hist: ", vendor, end=' ')
            et_guess = cross_check(et_guess, vend_hist[vendor], allowed_list, 7)
            if real_key in vend_hist[vendor]:
                in_vendor_history += 1
        if uvendor and uvendor in uvend_hist.keys():
            uvendor_history_exists += 1
            if pause_points:
                print("User Vendor History: ", uvendor, end=' ')
            et_guess = cross_check(et_guess, uvend_hist[uvendor], allowed_list, 7)
            if real_key in uvend_hist[uvendor]:
                in_uvendor_history += 1
        if userid in user_hist.keys():
            user_history_exists += 1
            if pause_points:
                print("User Hist: ", userid, end=' ')
            et_guess = cross_check(et_guess, user_hist[userid], allowed_list, 6)
            if real_key in user_hist[userid]:
                in_user_history += 1
        if company in comp_hist.keys():
            entity_history_exists += 1
            if pause_points:
                print("Entity Hist: ", company, end=' ')
            et_guess = cross_check(et_guess, comp_hist[company], allowed_list, 5)
            if real_key in comp_hist[company]:
                in_entity_history += 1
        if not len(et_guess):
            if company in all_entity_hist.keys():
                et_guess = cross_check(et_guess, all_entity_hist[company], allowed_list, 5)
            travel_words = ['trav', 't&', '&e', 'trans']
            for expkey, expname in allowed_list.items():
                if any([t in expname.lower() for t in travel_words]):
                    et_guess[expkey] += 1
        if not len(et_guess):
            undef += 1
            et_guess['UNDEF'] += 1
            print('allowed list: ', allowed_list)
            print(x_test[pred_index])
            print('real key: ', real_key)
            print('----------------------------')

        if pause_points:
            print("Final Guess rankings: ", et_guess)
        if et_guess.most_common(1)[0][0] == real_key:
            correct += 1

        if pause_points:
            print('Real ET:', real_key, allowed_list[real_key])
            total_counts = sum(et_guess.values())
            output = []
            for k, v in et_guess.most_common(5):
                output.append({'Expense Type': k, 'Score': float(v) / total_counts})
            print(total_counts, output)
            input('pause')

    print("ouput_name skipped: ", company_skipped)
    print("Real key not in allowed list: ", real_key_na, "(probably the entity changed their allowed types recently)")
    print("Total counter: ", total_counter)
    print("CORRECT ET GUESS: ", correct)
    print("# correct key somewhere in user hist / user history exists:", in_user_history, user_history_exists)
    print("# correct key somewhere in entity hist / entity history exists:", in_entity_history, entity_history_exists)
    print("# correct key somewhere in vendor hist / vendor history exists:", in_vendor_history, vendor_history_exists)
    print("# correct key somewhere in user-vendor hist / user-vendor hsitory exists:", in_uvendor_history, uvendor_history_exists)
    print("# undefs:", undef)

if __name__ == '__main__':
    remap()

