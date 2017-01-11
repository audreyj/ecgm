"""
Author: audreyc
Last Updated: 04/04/16

This file parses Everaldo's benchmark dataset, saved in DS_request format.
- Removes bad_chars from OCR text and replaces with the chars indicated
- Prints out one example result (last one)
- Saves out new pickle files to X_train, X_test, Y_train, and Y_test.  Ready for modeling.
"""

import pandas as pd
import json
import string
import DS_type_mapping
import pickle
from collections import Counter
from sklearn import cross_validation
import time


bad_chars = {'\\r': ' ', '\\n': ' ', '•': ' '}
             # '1': '•', '2': '•', '3': '•', '4': '•', '5': '•',
             # '6': '•', '7': '•', '8': '•', '9': '•', '0': '•'}
output_dir = 'sampledata2/'


def parse_input(file_name, cross_validate=True, ds_types=False):
    start = time.time()
    df = pd.read_csv(file_name, sep='\t')
    ds_suffix = 'DS' if ds_types else 'ET'
    type_order, type_dict = DS_type_mapping.query_file()

    doc_list = []
    target_list = []
    ds_target_list = []
    info_list = []
    user_hist = {}
    company_hist = {}
    vendor_hist = {}
    allowed_dict = {}
    count_violations = 0

    for i in df.index:

        sub_df = json.loads(df.ix[i, 'ds_request'])
        t = sub_df['ocrText']
        this_expense_key = df.ix[i, 'expense_type_legacy_key']
        this_expense_name = df.ix[i, 'expense_type_name']
        this_expense_name = ' '.join(word.strip(string.punctuation) for word in this_expense_name.split())
        this_expense_name = this_expense_name.lower()
        this_allowed_list = {x['ExpKey']: x['Name'] for x in sub_df['userExpenseTypes']}
        entity = sub_df['entityId']
        # if entity != 'p00425z4gu':
        #     continue
        user_key = sub_df['entityId'] + '-' + sub_df['userId']
        vendor = sub_df['vendor']

        if this_expense_key not in this_allowed_list.keys():
            # In this very rare case of violations, just drop...
            count_violations += 1
            # print("Not found: " + df.ix[i, 'expense_type_legacy_key'])  # + str(sub_df['userExpenseTypes']))
            # continue

        this_expense_ds_key = DS_type_mapping.to_ds_types(df.ix[i, 'expense_type_legacy_key'],
                                                          df.ix[i, 'expense_type_name'], type_order, type_dict)

        target_list.append(this_expense_key)
        ds_target_list.append(this_expense_ds_key)

        info_list.append({'datekey': df.ix[i, 'trans_date_key'], 'vendor': vendor,
                          'entity': entity, 'userid': sub_df['userId'],
                          'amount': sub_df['receiptAmt'], 'ds_request': sub_df})

        # This line for alpha characters only.  Actually it doesn't work very well.
        # t = re.sub('[^a-zA-Z \-]', ' ', df.ix[i, 'ocr_text'])

        # These lines are for replacing the bad characters (listed above) with some other token.
        for bc, rw in bad_chars.items():
            t = t.replace(bc, rw)
        # print(t)
        t = t.lower()
        s = t.split()
        s = [x for x in s if len(x) > 1]
        doc_list.append(' '.join(s))

        if i > 16812:
            continue

        if user_key not in user_hist.keys():
            user_hist[user_key] = Counter()
        user_hist[user_key][this_expense_key] += 1

        if entity not in company_hist.keys():
            company_hist[entity] = Counter()
        company_hist[entity][this_expense_key] += 1

        if vendor not in vendor_hist.keys():
            vendor_hist[vendor] = Counter()
        vendor_hist[vendor][this_expense_name] += 1

        if entity not in allowed_dict.keys():
            allowed_dict[entity] = this_allowed_list

    if cross_validate:
        # x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        #                                           doc_list, target_list, test_size=0.1, random_state=1)

        ninety = int(len(doc_list) * 0.9)
        print(ninety)
        x_train = doc_list[:ninety]
        x_test = doc_list[ninety:]
        y_train = target_list[:ninety]
        y_ds_train = ds_target_list[:ninety]
        y_test = target_list[ninety:]
        y_ds_test = ds_target_list[ninety:]
        info_train = info_list[:ninety]
        info_test = info_list[ninety:]

        pickle.dump(x_test, open(output_dir + 'sampled_X_test.pkl', 'wb'))
        pickle.dump(y_test, open(output_dir + 'sampled_Y_testET.pkl', 'wb'))
        pickle.dump(y_ds_test, open(output_dir + 'sampled_Y_testDS.pkl', 'wb'))
        pickle.dump(info_test, open(output_dir + 'sampled_info_test.pkl', 'wb'))
    else:
        x_train = doc_list
        y_train = target_list
        y_ds_train = ds_target_list
        info_train = info_list

    pickle.dump(x_train, open(output_dir + 'sampled_X_train.pkl', 'wb'))
    pickle.dump(y_train, open(output_dir + 'sampled_Y_trainET.pkl', 'wb'))
    pickle.dump(y_ds_train, open(output_dir + 'sampled_Y_trainDS.pkl', 'wb'))
    pickle.dump(info_train, open(output_dir + 'sampled_info_train.pkl', 'wb'))

    pickle.dump(vendor_hist, open(output_dir + 'sampled_vendorhist' + ds_suffix + '.pkl', 'wb'))
    pickle.dump(company_hist, open(output_dir + 'sampled_companyhist' + ds_suffix + '.pkl', 'wb'))
    pickle.dump(user_hist, open(output_dir + 'sampled_userhist' + ds_suffix + '.pkl', 'wb'))
    pickle.dump(allowed_dict, open(output_dir + 'sampled_allowedtypes.pkl', 'wb'))

    # print(len(doc_list))
    # print(len(target_list))
    # print(t)
    print(doc_list[0])
    print(target_list[0])
    print("violations skipped (expense key not in allowed list): " + str(count_violations))

    end = time.time()
    print("Time: %.2f seconds for %d entries" % (end-start, len(target_list)))

if __name__ == '__main__':
    input_file = 'C:/Users/audreyc/PyCharm/ecgm-new/new_ds_model/expenseit_token_benchmark_dataset.tsv'
    parse_input(input_file)
