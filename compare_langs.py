"""
author: audreyc
last updated: 01/29/2017
"""

import pickle
import json

file_name = 'error_analysis/exptype_eval_de'
with open(file_name, 'r', encoding='utf-8') as f:
    line_count = 0
    no_ds_response = 0
    no_user_history = 0
    no_expensetypes = 0
    total_assessed = 0
    et_correct = 0
    et_inresponse = 0
    et_same = 0
    for line in f:
        line_count += 1
        if line_count % 10000 == 0:
            print(line_count)
        p = line.split('\t')
        datekey = p[0]
        ds_req = p[1]
        ds_resp = p[2]
        location_country = p[3]
        expense_type_name = p[4]
        exp_key = p[5].replace('\n', '')

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

        # print("my exp", out_keys)
        # print(type(my_expkey), my_expkey)
        # print("real exp key", exp_key)
        # print(type(exp_key))
        # input("pause")

        if my_expkey == exp_key:
            et_correct += 1

    print("line count: " + str(line_count))
    print("no ds request: " + str(no_ds_response))
    print("no expense types in ds response: " + str(no_expensetypes))
    print("total assess: " + str(total_assessed))
    print("total et correct: %d (%.2f)" % (et_correct, et_correct / total_assessed * 100))
