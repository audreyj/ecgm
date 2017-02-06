#!/user/bin/env python

"""
author: audreyc
Last Update: 1/11/17
This file should take in a certain date-range of data from report entry and re-parse in RQA to get expense-type results
"""

import json
import sys
import os
import pickle
import DS_type_mapping
import time_extractor
from collections import Counter
import urllib.request


def parse_jan_file():
    jan_file = '../ECTokenEval/token-test-dsreq_2016-07-07_1534'
    total_line_count = 0
    no_ds_req = 0
    exp_types_dict = {}
    with open(jan_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_line_count += 1
            # if total_line_count > 10:
            #     break
            p = line.split('\t')
            entity = p[0]
            ds_req = p[12]
            if len(ds_req) < 2:
                no_ds_req += 1
                continue
            this_ds_req = json.loads(ds_req)
            if total_line_count == 1:
                print(this_ds_req['userExpenseTypes'])

            # print(ds_req)
            if entity not in exp_types_dict.keys():
                exp_types_dict[entity] = this_ds_req["userExpenseTypes"]

    print("total lines assessed: " + str(total_line_count))
    print("no ds request: " + str(no_ds_req))

    pickle.dump(exp_types_dict, open('error_analysis/entity_expense_types.pkl', 'wb'))


def process_all(file_name, exp_types):
    type_order, type_dict = DS_type_mapping.query_file()
    total_file_count = 0
    no_exp_types = 0
    date_is_stupid = 0
    ds_correct = 0
    et_correct = 0
    company_dict = {}
    company_types = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            total_file_count += 1
            # if total_file_count > 100:
            #     break
            p = line.split('\t')
            entity = p[0]
            userid = p[1]
            datekey = p[2]
            exp_key = p[3]
            exp_name = p[4]
            amount = p[5]
            vendor = p[6]
            ocr_text = p[7]

            if int(datekey) > 20170115:
                date_is_stupid += 1
                continue

            if entity not in exp_types.keys():
                no_exp_types += 1
                continue

            te = time_extractor.TimeExtractor()
            time_pttwo = te.extract_time(ocr_text)
            if time_pttwo and time_pttwo['time']:
                this_time = [{"value": time_pttwo['time'][0]['value'], "score": 1}]
            else:
                this_time = []

            ds_api_url = 'https://expense-type-data-sci-mspqaf5.concurasp.com/ETClassification'

            data = {"entityId": entity, "userId": str(userid), "debug": 1,
                    "amount": [{"value": float(amount), "score": 1}],
                    "time": this_time,
                    "currency": [{"value": "USD", "score": 1}],
                    "vendor": [{"value": vendor, "score": 1}],
                    "ocrText": ocr_text,
                    "userExpenseTypes": exp_types[entity]}

            data = json.dumps(data).encode('utf-8', 'ignore')

            req = urllib.request.Request(ds_api_url, data)
            kube_output = urllib.request.urlopen(req)
            t_str = kube_output.read().decode('utf-8')
            output = json.loads(t_str)

            # print(entity)
            # print(output['debug'][0])
            # print(output['expenseTypes'][0]['value'])
            # print(exp_key)
            # print('--------')

            if entity not in company_dict.keys():
                company_dict[entity] = Counter()
                company_types[entity] = {'num types': len(exp_types[entity]),
                                         'num meals': len([z for z in exp_types[entity]
                                                           if DS_type_mapping.to_ds_types(z['Name'],
                                                                type_order, type_dict) == 'MEALS'])}
            company_dict[entity]['total'] += 1

            # print(output['expenseTypes'][0]['type'], exp_key)

            if output['expenseTypes'][0]['value'] == exp_key:
                et_correct += 1
                company_dict[entity]['et correct'] += 1
            real_ds_type = DS_type_mapping.to_ds_types(exp_name, type_order, type_dict)
            if real_ds_type in output['debug'][0]:
                ds_correct += 1
                company_dict[entity]['ds correct'] += 1

    print("total file number: " + str(total_file_count))
    print("date is stupid: " + str(date_is_stupid))
    print("not in expense types pickle: " + str(no_exp_types))
    print("number DS correct: " + str(ds_correct))
    print("number ET correct: " + str(et_correct))

    outfile = open('error_analysis/DecemberTypes.txt', 'w+', encoding='utf-8')
    outfile.write("Company\tDS Correct\tET Correct\tCount\tNum Types\tNum MealTypes\n")
    for c in company_dict.keys():
        outfile.write(c + '\t')
        outfile.write(str(company_dict[c]['ds correct']) + '\t')
        outfile.write(str(company_dict[c]['et correct']) + '\t')
        outfile.write(str(company_dict[c]['total']) + '\t')
        outfile.write(str(company_types[c]['num types']) + '\t')
        outfile.write(str(company_types[c]['num meals']) + '\n')


if __name__ == '__main__':
    if not os.path.isfile('error_analysis/entity_expense_types.pkl'):
        parse_jan_file()
    exp_types = pickle.load(open('error_analysis/entity_expense_types.pkl', 'rb'))
    dec_file = 'error_analysis/decset'
    process_all(dec_file, exp_types)

