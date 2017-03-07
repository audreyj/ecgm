"""
author: audreyc
last updated: 01/29/2017
"""

import pickle
import json
import os
import sys
import numpy as np
import time_extractor
import urllib.request
import matplotlib.pyplot as plt
from collections import Counter
import DS_type_mapping

pause_points = True
concur_expense_types = {"userExpenseTypes": [
      {  "ExpKey":"AIRFR", "Name":"Airfare", "SpdCat":"OTHER"  },
      {  "ExpKey":"01194", "Name":"Insurance", "SpdCat":"OTHER"  },
      {  "ExpKey":"00005", "Name":"Books and Reference Material", "SpdCat":"OTHER"  },
      {  "ExpKey":"00008", "Name":"Internet Access", "SpdCat":"OTHER"  },
      {  "ExpKey":"00010", "Name":"Hotel Phone",  "SpdCat":"OTHER"  },
      {  "ExpKey":"00013", "Name":"Miscellaneous Expense", "SpdCat":"OTHER" },
      {  "ExpKey":"00016", "Name":"Software", "SpdCat":"OTHER"  },
      {  "ExpKey":"00017", "Name":"Shipping", "SpdCat":"OTHER"  },
      {  "ExpKey":"00030", "Name":"American Express Fees", "SpdCat":"OTHER"  },
      {  "ExpKey":"00031", "Name":"Tips", "SpdCat":"OTHER"  },
      {  "ExpKey":"00033", "Name":"Other Travel Expenses", "SpdCat":"OTHER"  },
      {  "ExpKey":"00090", "Name":"Business Meals - Meetings", "SpdCat":"MEALN"  },
      {  "ExpKey":"00091", "Name":"Company - Employee Events",  "SpdCat":"OTHER" },
      {  "ExpKey":"00092", "Name":"Conf - Seminar - Trng", "SpdCat":"OTHER" },
      {  "ExpKey":"00094", "Name":"Parking - Tolls", "SpdCat":"OTHER"  },
      {  "ExpKey":"00095", "Name":"Publications - Subscriptions", "SpdCat":"OTHER"  },
      {  "ExpKey":"00100", "Name":"Equipment.", "SpdCat":"OFFIC"  },
      {  "ExpKey":"00101", "Name":"Taxi-Shuttle-Train", "SpdCat":"OTHER"  },
      {  "ExpKey":"00110", "Name":"Booking Fees", "SpdCat":"OTHER"  },
      {  "ExpKey":"01130", "Name":"Website Fees", "SpdCat":"OTHER"  },
      {  "ExpKey":"01140", "Name":"Congestion Charge", "SpdCat":"OTHER"  },
      {  "ExpKey":"01141", "Name":"Entertainment - Other", "SpdCat":"OTHER"  },
      {  "ExpKey":"01142", "Name":"Entertainment - Staff", "SpdCat":"OTHER"  },
      {  "ExpKey":"01143", "Name":"Home Business Line", "SpdCat":"OTHER"  },
      {  "ExpKey":"01144", "Name":"Gifts (Non-employee)", "SpdCat":"OTHER"  },
      {  "ExpKey":"01150", "Name":"Subsistence Meal (>1 Employee)", "SpdCat":"OTHER"  },
      {  "ExpKey":"01151", "Name":"Internet Access - Travel", "SpdCat":"OTHER"  },
      {  "ExpKey":"01171", "Name":"Airfare Fees", "SpdCat":"OTHER"  },
      {  "ExpKey":"01181", "Name":"Beverages", "SpdCat":"MEALA"  },
      {  "ExpKey":"BRKFT", "Name":"Individual Breakfast", "SpdCat":"MEALA"  },
      {  "ExpKey":"CARRT", "Name":"Car Rental", "SpdCat":"OTHER"  },
      {  "ExpKey":"CELPH", "Name":"Cellular - Mobile Phone", "SpdCat":"OTHER"      },
      {  "ExpKey":"DINNR", "Name":"Individual Dinner", "SpdCat":"MEALA"  },
      {  "ExpKey":"DUESX", "Name":"Membership Dues", "SpdCat":"OTHER"  },
      {  "ExpKey":"ENTOT", "Name":"Entertainment", "SpdCat":"MEALA"  },
      {  "ExpKey":"FAXXX", "Name":"Fax", "SpdCat":"OTHER"  },
      {  "ExpKey":"GASXX", "Name":"Gas - Petrol (rental car only)", "SpdCat":"OTHER"  },
      {  "ExpKey":"GIFTS", "Name":"Gifts - Incentives (Employee)",  "SpdCat":"OTHER"  },
      {  "ExpKey":"HOMPH", "Name":"Local Phone", "SpdCat":"OTHER"  },
      {  "ExpKey":"JTRAN", "Name":"Japan Public Transportation", "SpdCat":"JGTRN"  },
      {  "ExpKey":"LODNG", "Name":"Hotel", "SpdCat":"OTHER"  },
      {  "ExpKey":"LUNCH", "Name":"Individual Lunch", "SpdCat":"MEALA"  },
      {  "ExpKey":"MILEG", "Name":"Mileage (personal car only)", "SpdCat":"PRCRM"  },
      {  "ExpKey":"OFCSP", "Name":"Office Supplies", "SpdCat":"OTHER"  },
      {  "ExpKey":"POSTG", "Name":"Postage", "SpdCat":"OTHER"  },
      {  "ExpKey":"SEMNR", "Name":"Printing Expenses", "SpdCat":"OTHER"  },
      {  "ExpKey":"TELPH", "Name":"Long Distance", "SpdCat":"OTHER"  },
      {  "ExpKey":"TRDSH", "Name":"Trade Shows", "SpdCat":"OTHER"  },
      {  "ExpKey":"UNDEF", "Name":"Undefined", "SpdCat":"OTHER"  },
      {  "ExpKey":"01201", "Name":"Collateral/Sales Tools", "SpdCat":"OTHER"  },
      {  "ExpKey":"01202", "Name":"Expatriate Employee Expenses", "SpdCat":"OTHER"  },
      {  "ExpKey":"01203", "Name":"Parking Subsidy (Bellevue Employees Only)", "SpdCat":"OTHER"  },
      {  "ExpKey":"01197", "Name":"Beverages - Alcohol", "SpdCat":"OTHER"  },
      {  "ExpKey":"01200", "Name":"Marketing Events ", "SpdCat":"OTHER"  },
      {  "ExpKey":"LNDRY", "Name":"Laundry", "SpdCat":"OTHER"  }
   ]}


class ErrorAnalyzer:
    def __init__(self):
        self.line_count = 0

        self.entity = ''
        self.userid = 0
        self.amount = 0.0
        self.currency = 'USD'
        self.location_country = ''
        self.datekey = 0
        self.exp_key = ''
        self.exp_name = ''
        self.time = 0
        self.ocr_text = ''
        self.vendor = ''

        self.all_exp_types_dict = {}
        self.smb_dict = {}
        self.entity_histories_dict = {}
        self.user_histories_dict = {}

        self.skip_reasons = {'no exp types': 0, 'date is stupid': 0, 'no ds request': 0}
        self.my_exp_key = ''

    def process_skip(self):
        skip_command = 0

        # if self.location_country not in ['FR', 'DE']:
        #     skip_command = 1
        if self.entity != 'p0055084fh8j':
            skip_command = 1
        # if userid not in user_search:
        #     skip_command = 1
        # if self.entity not in self.all_exp_types_dict.keys():
        #     self.skip_reasons['no exp types'] += 1
        #     skip_command = 1

        if int(self.datekey) > 20170115:
            self.skip_reasons['date is stupid'] += 1
            skip_command = 1
        if self.my_exp_key == 'skipped':
            self.skip_reasons['no ds request'] += 1
            skip_command = 1
        return skip_command

    def parse_jan_file(self):
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

    def parse_client_file(self):
        client_file = '../company_segments/client_salesforce.csv'
        total_line_count = 0
        no_smb_cat = 0
        smb_dict = {}
        with open(client_file, 'r', encoding='utf-16') as f:
            for line in f:
                # print(line)
                total_line_count += 1
                if total_line_count == 1:
                    continue
                p = line.split('\t')
                entity_name = p[0]
                entity_size = p[1]
                entity_id = p[2]
                smb_cat = p[6]
                if len(smb_cat) < 2:
                    no_smb_cat += 1
                    continue

                if entity_id not in smb_dict.keys():
                    smb_dict[entity_id] = {'Name': entity_name, 'Size': entity_size, 'Category': smb_cat}

        print("total lines assessed: " + str(total_line_count))
        print("no smb category: " + str(no_smb_cat))

        pickle.dump(smb_dict, open('error_analysis/entity_smb.pkl', 'wb'))

    def call_api(self):
        ds_api_url = 'https://expense-type-data-sci-seaintf5.concurasp.com/ETClassification'
        header = {'concur-correlationid': 'audrey-test'}

        te = time_extractor.TimeExtractor()
        time_pttwo = te.extract_time(self.ocr_text)
        if time_pttwo and time_pttwo['time']:
            self.time = [{"value": time_pttwo['time'][0]['value'], "score": 1}]
        else:
            self.time = []

        data = {"entityId": str(self.entity), "userId": str(self.userid), "debug": 1,
                "amount": [{"value": float(self.amount), "score": 1}],
                "time": self.time,
                "currency": [{"value": self.currency, "score": 1}],
                "vendor": [{"value": self.vendor, "score": 1}],
                "ocrText": self.ocr_text,
                "userExpenseTypes": self.all_exp_types_dict[self.entity]}

        data = json.dumps(data).encode('utf-8', 'ignore')

        req = urllib.request.Request(ds_api_url, data, headers=header)
        kube_output = urllib.request.urlopen(req)
        t_str = kube_output.read().decode('utf-8')
        output_dict = json.loads(t_str)
        return output_dict

    def parse_input_line(self, p, file_type=0):
        # file_type = 0: audrey's version
        # file_type = 1: mike's version
        if file_type == 0:
            self.entity = p[0]
            self.userid = p[1]
            self.datekey = p[2]
            self.exp_key = p[3]
            self.exp_name = p[4]
            self.amount = p[5]
            self.currency = 'USD'
            self.vendor = p[6]
            self.ocr_text = p[7]

        elif file_type == 1:
            self.entity = p[0]
            self.userid = p[1]
            imageid = p[2]
            self.datekey = p[3]
            self.currency = p[4]
            self.amount = p[5]
            location_city = p[6]
            location_state = p[7]
            self.location_country = p[8]
            self.vendor = p[9]
            self.exp_name = p[10]
            self.exp_key = p[11]
            ds_req = p[12]

            if len(ds_req) > 2:
                self.my_exp_key = 'fetch'
                ds_request = json.loads(ds_req)
                self.all_exp_types_dict[self.entity] = ds_request['userExpenseTypes']
                self.ocr_text = ds_request['ocrText']
                if not isinstance(self.ocr_text, str):
                    self.ocr_text = ''
                    # print('ocr text not a string')
            else:
                self.my_exp_key = 'skipped'

    def count_histories(self, text_list):
        # search_list = ['cb: [exp-vendor]', 'cb: [exp-amount]', 'cb: [exp-user]', 'cb: [exp-entity]']
        user_hash = str(self.userid) + '-' + str(self.entity)
        for t in text_list['debug']:
            if t.startswith('cb: [exp-entity]') or t.startswith('cb: [exp-user]'):
                parts = t.split(' => ')
                if pause_points:
                    print(t)
                    # print(parts[0].count('-'))
                    # print(parts[1])
                if parts[1] != 'None':
                    history_dict = json.loads(parts[1].replace("'", '"'))
                    if t.startswith('cb: [exp-entity]'):
                        self.entity_histories_dict[self.entity] = sum(history_dict.values())
                    elif t.startswith('cb: [exp-user]'):
                        self.user_histories_dict[user_hash] = sum(history_dict.values())

    def write_out_files(self, company_dict, type_order, type_dict):
        outfile = open('error_analysis/DecemberTypes.txt', 'w+', encoding='utf-8')
        outfile.write("Company\tET Correct\tCount\tEntity Hist Length\tNum Types\t")
        outfile.write("Num Meals\tNum Employees\tCompany Name\tCompany Cat\n")
        for c in company_dict.keys():
            outfile.write(c + '\t')
            outfile.write(str(company_dict[c]['et correct']) + '\t')
            outfile.write(str(company_dict[c]['total']) + '\t')
            if c in self.entity_histories_dict.keys():
                outfile.write(str(self.entity_histories_dict[c]) + '\t')
            else:
                outfile.write('0\t')
            meal_types = [z for z in self.all_exp_types_dict[c] if
                          DS_type_mapping.to_ds_types(z['Name'], type_order, type_dict) == 'MEALS']
            outfile.write(str(len(self.all_exp_types_dict[c])) + '\t')
            outfile.write(str(len(meal_types)) + '\t')
            if c in self.smb_dict.keys():
                outfile.write(str(self.smb_dict[c]['Size']) + '\t')
                outfile.write(str(self.smb_dict[c]['Name']) + '\t')
                outfile.write(str(self.smb_dict[c]['Category']) + '\n')
            else:
                outfile.write('-\t-\t-\n')

        pickle.dump(self.smb_dict, open('error_analysis/smb_dict.pkl', 'wb'))
        pickle.dump(self.user_histories_dict, open('error_analysis/user_histories.pkl', 'wb'))
        pickle.dump(self.all_exp_types_dict, open('error_analysis/entity_expense_types.pkl', 'wb'))
        pickle.dump(self.entity_histories_dict, open('error_analysis/entity_histories.pkl', 'wb'))

    def get_data(self, file_name, file_type_num, max_lines):
        """
        this cycles through the file (up to max_lines) and counts instances of everything into a dict
        """
        type_order, type_dict = DS_type_mapping.query_file()
        # user_search = ['28177', '611', '30540']
        total_assessed = 0
        et_correct = 0
        company_dict = {}
        company_types = {}
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                self.line_count += 1
                if max_lines and self.line_count > max_lines:
                    break
                if self.line_count % 10000 == 0:
                    print(self.line_count, self.skip_reasons)
                p = line.split('\t')
                self.parse_input_line(p, file_type_num)

                if self.process_skip():
                    continue

                total_assessed += 1

                output = self.call_api()
                self.my_exp_key = output['expenseTypes'][0]['value']
                self.count_histories(output)

                if self.entity not in company_dict.keys():
                    company_dict[self.entity] = Counter()
                    company_types[self.entity] = Counter()
                company_dict[self.entity]['total'] += 1
                company_types[self.entity][self.exp_key] += 1

                # print(output['expenseTypes'][0]['type'], exp_key)

                if self.my_exp_key == self.exp_key:
                    et_correct += 1
                    company_dict[self.entity]['et correct'] += 1

                if pause_points:
                    for et in self.all_exp_types_dict[self.entity]:
                        print(et['ExpKey'], ', ', et['Name'])
                    print("Number of expense types: ", len(self.all_exp_types_dict[self.entity]))
                    print("My output: ", output['expenseTypes'])
                    print("Real Expense Type: %s, %s" % (self.exp_key, self.exp_name))
                    print(self.vendor)
                    print(self.ocr_text)
                    input('----- pause ------')

        print("total lines: %d" % self.line_count)
        print(self.skip_reasons)
        print("total assessed: %d" % total_assessed)
        print("total et correct: %d" % et_correct)

        # self.write_out_files(company_dict, type_order, type_dict)
        print(company_dict[self.entity])

    def run_all(self, file_name, file_type='audrey', max_lines=0):
        """
        this runs each piece in the correct order, with just the filename and max_lines as inputs
        I broke into these pieces because I might be able to use each piece individually
        """
        # if not os.path.isfile('error_analysis/entity_expense_types.pkl'):
        #     self.parse_jan_file()
        # self.all_exp_types_dict = pickle.load(open('error_analysis/entity_expense_types.pkl', 'rb'))
        if not os.path.isfile('error_analysis/entity_smb.pkl'):
            self.parse_client_file()
        self.smb_dict = pickle.load(open('error_analysis/entity_smb.pkl', 'rb'))

        file_type_num = 0
        if file_type == 'audrey':
            file_type_num = 0
        elif file_type == 'mike':
            file_type_num = 1

        self.get_data(file_name, file_type_num, max_lines)


if __name__ == "__main__":
    EA = ErrorAnalyzer()
    # run_all(sys.argv[1])
    # run_all('sampledata2/sampled_Y_trainDS.pkl')
    # run_all('C:/Users/audreyc/PyCharm/octset2')
    # EA.run_all('error_analysis/decset')
    EA.run_all('error_analysis/dsreq-201701', 'mike')
