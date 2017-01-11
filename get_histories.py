"""
author: audreyc
Last Update: 05/02/16
Gets user, entity, and amount history out of large dataset and puts it into inputdata/
Updated 05/02 for move to all-companies on compute machine
"""

import sys
import DS_type_mapping
import numpy as np
import pickle
from collections import Counter
import time

company_list = ['p0000745jr8u', 'p0003884x7lt', 'p0043611aoji', 'p0039557fvbf',
                'p0090051h2oq', 'p0006679vz2s', 'p00425z4gu', 'p0009976ed7k',
                'p0005859huep', 'p0089280mxzg', 'p0079383unbs', 'p00521862bvn',
                'p0002406hlem', 'p0002233vdsm', 'p0096373arhm', 'p0039096va39',
                'p0046598vm29', 'p0036394dcvf', 'p0081759udz2', 'p0079434mdig',
                'p0014655asb9', 'p0043883wrzh']


def parse_input(company):
    start = time.time()

    file_name = 'inputdata/' + company + '_trainset'

    user_hist = {}
    company_hist = {}
    amount_hist = {}
    amount_avg = {}
    line_number = 0

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            p = line.split('\t')
            entity = p[0]
            userid = p[1]
            datekey = int(p[2])
            expense_key = p[3]
            expense_name = p[4]
            longer_type = True if len(p) > 6 else False
            if not longer_type:
                ocr = p[5]
            else:
                amount = p[5]
                vendor = p[6]
                expenseit = p[7]
                ocr = p[8]

            user_key = entity + '-' + userid
            full_expense = expense_key + '|' + expense_name

            if user_key not in user_hist.keys():
                user_hist[user_key] = Counter()
            user_hist[user_key][full_expense] += 1

            if entity not in company_hist.keys():
                company_hist[entity] = Counter()
            company_hist[entity][full_expense] += 1

            if not longer_type:
                continue
            if float(amount) == 0:
                entity_amount = entity + '-000'
            elif float(amount) < 10:
                entity_amount = entity + '-001'
            elif float(amount) > 100:
                entity_amount = entity + '-100'
            else:
                entity_amount = entity + '-010'
            if entity_amount not in amount_hist.keys():
                amount_hist[entity_amount] = Counter()
            amount_hist[entity_amount][full_expense] += 1

            if full_expense not in amount_avg.keys():
                amount_avg[full_expense] = []
            amount_avg[full_expense].append(float(amount))

    type_amount = {}
    for k, v in amount_avg.items():
        if len(v) < 10:
            new_v = v
        else:
            new_v = [x for x in v if np.percentile(v, 90) > x > np.percentile(v, 10)]
        if len(new_v) == 0:
            type_amount[k] = 0
        else:
            type_amount[k] = [np.average(new_v), np.std(new_v), len(v)]
    pickle.dump(company_hist, open('inputdata/' + company + '_companyhist.pkl', 'wb'))
    pickle.dump(user_hist, open('inputdata/' + company + '_userhist.pkl', 'wb'))
    pickle.dump(amount_hist, open('inputdata/' + company + '_amounthist.pkl', 'wb'))
    pickle.dump(type_amount, open('inputdata/' + company + '_typeamount.pkl', 'wb'))
    end = time.time()
    print("Time: %.2f seconds for %d entries" % (end-start, line_number))

if __name__ == '__main__':
    # input_file = 'inputdata/ge_trainset'
    for company in company_list:
        print("========== Entity: %s ===========" % company)
        parse_input(company)
