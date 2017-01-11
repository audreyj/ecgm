"""
author: audreyc
updated: 08/22/2016
Creates history and training files from january testset (with ds requests)
Creates test files from benchmark dataset
"""

import pandas as pd
import json
import string
import DS_type_mapping
import pickle
from collections import Counter
import time


bad_chars = {'\\r': ' ', '\\n': ' ', '•': ' '}
output_dir = 'sampledata2/'


class SetMaker:
    def __init__(self):
        self.jan_file = 'token-test-dsreq_2016-07-07_1534'
        self.benchmark_file = 'new_ds_model/expenseit_token_benchmark_dataset.tsv'
        self.start_time = time.time()
        self.type_order, self.type_dict = DS_type_mapping.query_file()

    def train_data(self):
        total_file_count = 0
        no_ds_request = 0
        count_violations = 0
        x_train = []
        y_train = []
        y_ds_train = []
        info_train = []
        with open(self.jan_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_file_count += 1
                if total_file_count == 1:
                    continue
                # if total_file_count > 1000:
                #     break
                p = line.split('\t')
                entity = p[0]
                userid = p[1]
                exp_name = p[10]
                vendor = p[9]
                exp_key = p[11]
                ds_req = p[12]
                datekey = p[3]

                if len(ds_req) < 2:
                    no_ds_request += 1
                    continue
                data_input = json.loads(ds_req)
                t = data_input['ocrText']

                this_allowed_list = {x['ExpKey']: x['Name'] for x in data_input['userExpenseTypes']}

                if exp_key not in this_allowed_list.keys():
                    # In this very rare case of violations, just drop...
                    count_violations += 1
                    # continue

                this_expense_ds_key = DS_type_mapping.to_ds_types(exp_name,
                                                                  self.type_order, self.type_dict)

                y_ds_train.append(this_expense_ds_key)
                y_train.append(exp_key)

                info_train.append({'datekey': datekey, 'vendor': vendor,
                                  'entity': entity, 'userid': data_input['userId'],
                                  'amount': data_input['receiptAmt'], 'ds_request': data_input})

                for bc, rw in bad_chars.items():
                    t = t.replace(bc, rw)
                t = t.lower()
                s = t.split()
                s = [x for x in s if len(x) > 1]
                x_train.append(' '.join(s))

        pickle.dump(x_train, open(output_dir + 'sampled_X_train.pkl', 'wb'))
        pickle.dump(y_train, open(output_dir + 'sampled_Y_trainET.pkl', 'wb'))
        pickle.dump(y_ds_train, open(output_dir + 'sampled_Y_trainDS.pkl', 'wb'))
        pickle.dump(info_train, open(output_dir + 'sampled_info_train.pkl', 'wb'))
        print(x_train[0])
        print(y_train[0], y_ds_train[0])
        print("no ds request: ", no_ds_request)
        print("total lines read: ", total_file_count)
        print("violations not skipped (expense key not in allowed list): " + str(count_violations))

    def test_data(self):
        df = pd.read_csv(self.benchmark_file, sep='\t')
        x_test = []
        y_test = []
        y_ds_test = []
        info_test = []
        allowed_dict = {}
        total_file_count = 0
        count_violations = 0
        for i in df.index:
            total_file_count += 1
            sub_df = json.loads(df.ix[i, 'ds_request'])
            t = sub_df['ocrText']
            this_expense_key = df.ix[i, 'expense_type_legacy_key']
            this_expense_name = df.ix[i, 'expense_type_name']
            this_allowed_list = {x['ExpKey']: x['Name'] for x in sub_df['userExpenseTypes']}
            entity = sub_df['entityId']
            user_key = sub_df['entityId'] + '-' + sub_df['userId']
            vendor = sub_df['vendor']

            if this_expense_key not in this_allowed_list.keys():
                # In this very rare case of violations, just drop...
                count_violations += 1
                # continue

            this_expense_ds_key = DS_type_mapping.to_ds_types(this_expense_name,
                                                              self.type_order, self.type_dict)

            y_test.append(this_expense_key)
            y_ds_test.append(this_expense_ds_key)

            info_test.append({'datekey': df.ix[i, 'trans_date_key'], 'vendor': vendor,
                              'entity': entity, 'userid': sub_df['userId'],
                              'amount': sub_df['receiptAmt'],
                              'expense_type': df.ix[i, 'expense_type_legacy_key']})

            # These lines are for replacing the bad characters (listed above) with some other token.
            for bc, rw in bad_chars.items():
                t = t.replace(bc, rw)
            t = t.lower()
            s = t.split()
            s = [x for x in s if len(x) > 1]
            x_test.append(' '.join(s))

            if entity not in allowed_dict.keys():
                allowed_dict[entity] = this_allowed_list

        pickle.dump(x_test, open(output_dir + 'sampled_X_test.pkl', 'wb'))
        pickle.dump(y_test, open(output_dir + 'sampled_Y_testET.pkl', 'wb'))
        pickle.dump(y_ds_test, open(output_dir + 'sampled_Y_testDS.pkl', 'wb'))
        pickle.dump(info_test, open(output_dir + 'sampled_info_test.pkl', 'wb'))
        pickle.dump(allowed_dict, open(output_dir + 'sampled_allowedtypes.pkl', 'wb'))

        print(x_test[0])
        print(y_test[0], y_ds_test[0])
        print("total lines read: ", total_file_count)
        print("violations not skipped (expense key not in allowed list): " + str(count_violations))


if __name__ == '__main__':
    test = SetMaker()
    test.train_data()
    test.test_data()
    end = time.time()
    print("Done.  Time: ", end-test.start_time)
