"""
author: audreyc
Last Update: 04/28/16
My goal in gen_files is to move through the large file only once.
"""

import time
import numpy as np
import pickle
from collections import Counter

# company_list = ['p0000745jr8u', 'p0003884x7lt', 'p0043611aoji', 'p0039557fvbf',
#                 'p0090051h2oq', 'p0006679vz2s', 'p00425z4gu', 'p0009976ed7k',
#                 'p0005859huep', 'p0089280mxzg', 'p0079383unbs', 'p00521862bvn',
#                 'p0002406hlem', 'p0002233vdsm', 'p0096373arhm', 'p0039096va39',
#                 'p0046598vm29', 'p0036394dcvf', 'p0081759udz2', 'p0079434mdig',
#                 'p0014655asb9', 'p0043883wrzh']
company_list = pickle.load(open('inputdata/all_company_list.pkl', 'rb'))


def gen_files():
    start = time.time()
    file_name = 'inputdata/all_trainset'
    # files = {c_name: open('inputdata/' + c_name + '_trainset', 'w') for c_name in company_list}
    userid_lookup = pickle.load(open('inputdata/userid_dict.pkl', 'rb'))

    user_hist = {}
    company_hist = {}
    vendor_hist = {}
    amount_avg = {}
    amount_hist = {}
    missing_userid = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.split('\t')
            entity = p[0]
            # if entity in company_list:
            #     with open('companydata/' + entity + '_trainset', 'a') as company_file:
            #         company_file.write(line)

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
                ocr = p[7]

            employee_hash = entity + '-' + userid
            if employee_hash not in userid_lookup.keys():
                missing_userid += 1
                continue
            user_key = userid_lookup[employee_hash]
            full_expense = expense_key + '|' + expense_name

            if user_key not in user_hist.keys():
                user_hist[user_key] = Counter()
            user_hist[user_key][full_expense] += 1

            if entity not in company_hist.keys():
                company_hist[entity] = Counter()
            company_hist[entity][full_expense] += 1

            if vendor:
                vendor_txt = vendor.replace(' ', '').lower()
                vendor_key = str(user_key) + '-' + vendor_txt
                if vendor_key not in vendor_hist.keys():
                    vendor_hist[vendor_key] = Counter()
                vendor_hist[vendor_key][full_expense] += 1

            if amount:
                amount = float(amount)
                if entity not in amount_hist.keys():
                    amount_hist[entity] = {}
                if 'Zero' not in amount_hist[entity].keys():
                    amount_hist[entity]['Zero'] = Counter()
                    amount_hist[entity]['Under Ten'] = Counter()
                    amount_hist[entity]['Under Fifty'] = Counter()
                    amount_hist[entity]['Under Hundred'] = Counter()
                    amount_hist[entity]['Under Thousand'] = Counter()
                    amount_hist[entity]['Other'] = Counter()
                if amount == 0:
                    amount_hist[entity]['Zero'][full_expense] += 1
                elif amount < 10:
                    amount_hist[entity]['Under Ten'][full_expense] += 1
                elif amount < 50:
                    amount_hist[entity]['Under Fifty'][full_expense] += 1
                elif amount < 100:
                    amount_hist[entity]['Under Hundred'][full_expense] += 1
                elif amount < 1000:
                    amount_hist[entity]['Under Thousand'][full_expense] += 1
                else:
                    amount_hist[entity]['Other'][full_expense] += 1

    # type_amount = {}
    # for k, v in amount_avg.items():
    #     if len(v) < 10:
    #         new_v = v
    #     else:
    #         new_v = [x for x in v if np.percentile(v, 90) > x > np.percentile(v, 10)]
    #     if len(new_v) == 0:
    #         type_amount[k] = 0
    #     else:
    #         type_amount[k] = [np.average(new_v), np.std(new_v), len(v)]

    master_remove_list = []
    for v_key, v_value in vendor_hist.items():
        inner_remove_list = []
        for e_key, e_value in v_value.items():
            if e_value < 2:
                inner_remove_list.append(e_key)
        for r_key in inner_remove_list:
            del vendor_hist[v_key][r_key]
        if not vendor_hist[v_key].keys():
            master_remove_list.append(v_key)
    for m_key in master_remove_list:
        del vendor_hist[m_key]

    pickle.dump(company_hist, open('inputdata/all_companyhist.pkl', 'wb'))
    pickle.dump(user_hist, open('inputdata/all_userhist.pkl', 'wb'))
    pickle.dump(vendor_hist, open('inputdata/all_vendorhist.pkl', 'wb'))
    # pickle.dump(type_amount, open('inputdata/all_typeamount.pkl', 'wb'))
    pickle.dump(amount_hist, open('inputdata/all_amounthist.pkl', 'wb'))
    end = time.time()
    print("missing user ids: ", str(missing_userid))
    print("Time: %.2f seconds" % (end - start))

if __name__ == '__main__':
    gen_files()
