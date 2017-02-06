"""
author: audreyc
last updated: 01/29/2017
"""

import pickle
import json

ranges = [0, 10, 20, 50, 100, 200, 400, 800, 1000]
output = []
for m, n in enumerate(ranges):
    if n == 0:
        continue
    prev = ranges[m-1]
    output.append({'Name': str(prev) + '-' + str(n), 'Count': 0, 'Correct' : 0})
output.append({'Name': str(ranges[-1]) + '+', 'Count': 0, 'Correct': 0})

user_history_dict = pickle.load(open('error_analysis/user_histories.pkl', 'rb'))
# for t, k in user_history_dict.items():
#     print(t, k)
#     input("pause")
file_name = 'error_analysis/dsresp-201701'
with open(file_name, 'r', encoding='utf-8') as f:
    line_count = 0
    no_ds_response = 0
    no_user_history = 0
    no_expensetypes = 0
    total_assessed = 0
    et_correct = 0
    et_inresponse = 0
    for line in f:
        line_count += 1
        if line_count % 10000 == 0:
            print(line_count)
        p = line.split('\t')
        entity = p[0]
        userid = p[1]
        imageid = p[2]
        datekey = p[3]
        currency = p[4]
        amount = p[5]
        location_city = p[6]
        location_state = p[7]
        location_country = p[8]
        vendor = p[9]
        exp_name = p[10]
        exp_key = p[11]
        ds_resp = p[12]

        if len(ds_resp) > 2:
            ds_response = json.loads(ds_resp)
        else:
            no_ds_response += 1
            continue

        if 'expenseTypes' not in ds_response['tokensV2'].keys():
            no_expensetypes += 1
            continue

        total_assessed += 1
        out_keys = ds_response['tokensV2']['expenseTypes']
        my_expkey = out_keys[0]['value']
        topkeys = []
        for k in out_keys:
            topkeys.append(k['value'])

        user_hash = str(userid) + '-' + str(entity)
        if user_hash not in user_history_dict:
            no_user_history += 1
            continue
        for i, r in enumerate(ranges):
            if r == 0:
                continue
            if ranges[i-1] < user_history_dict[user_hash] < r:
                output[i-1]['Count'] += 1
                if my_expkey == exp_key:
                    output[i-1]['Correct'] += 1
        if user_history_dict[user_hash] > ranges[-1]:
            output[-1]['Count'] += 1
            if my_expkey == exp_key:
                output[-1]['Correct'] += 1

        # print(user_history_dict[user_hash])
        # print("my exp", out_keys)
        # print("real exp key", exp_key)
        # for o in output:
        #     print(o)
        # input("pause")

        if my_expkey == exp_key:
            et_correct += 1
        if my_expkey in topkeys:
            et_inresponse += 1

    print("line count: " + str(line_count))
    print("no ds request: " + str(no_ds_response))
    print("no expense types in ds response: " + str(no_expensetypes))
    print("total assess: " + str(total_assessed))
    print("total et correct: %d (%.2f)" % (et_correct, et_correct / total_assessed * 100))
    print("total et incorrect but in top 5: %d (%.2f)" % (et_inresponse, et_inresponse / total_assessed * 100))
    print("no user history: " + str(no_user_history))
    for o in output:
        print(o)
