#!/user/bin/env python

"""
author: audreyc
Last Update: 7/19/2016
"""

import json
import sys
import pickle
import DS_type_mapping
import time_extractor
from collections import Counter
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
import re
import urllib.request


class Remapper:
    def __init__(self):
        # load history files
        self.user_hist = pickle.load(open('new_ds_model/all_userhist.pkl', 'rb'))
        self.comp_hist = pickle.load(open('new_ds_model/all_companyhist.pkl', 'rb'))
        self.vend_hist = pickle.load(open('new_ds_model/all_vendorhist.pkl', 'rb'))
        self.amt_hist = pickle.load(open('inputdata/all_amounthist.pkl', 'rb'))
        self.type_order, self.type_dict = DS_type_mapping.query_file()

    def get_history(self, entity, user, vend, amount):
        if entity in self.comp_hist.keys():
            comphist = self.comp_hist[entity]
        else:
            comphist = []
        if user in self.user_hist.keys():
            userhist = self.user_hist[user]
        else:
            userhist = []
        if vend in self.vend_hist.keys():
            vendhist = self.vend_hist[vend]
        else:
            vendhist = []
        if entity in self.amt_hist.keys() and amount:
            if amount == 0:
                amthist = self.amt_hist[entity]['Zero']
            elif amount < 10:
                amthist = self.amt_hist[entity]['Under Ten']
            elif amount < 50:
                amthist = self.amt_hist[entity]['Under Fifty']
            elif amount < 100:
                amthist = self.amt_hist[entity]['Under Hundred']
            elif amount < 1000:
                amthist = self.amt_hist[entity]['Under Thousand']
            else:
                amthist = self.amt_hist[entity]['Other']
        else:
            amthist = []
        return userhist, comphist, vendhist, amthist

    def cross_check(self, ct, history_counter, allowed_list, max_pts=5):
        ordered_names_all = sorted(history_counter, key=history_counter.get, reverse=True)
        ordered_names = [y for y in ordered_names_all if y in allowed_list.keys()]
        if max_pts >= 10:
            multiplier = 2
            max_pts = int(max_pts / 2)
        else:
            multiplier = 1
        for e_num, et_key in enumerate(ordered_names):
            if e_num >= max_pts:
                break
            ct[et_key] += (max_pts - e_num) * multiplier
        return ct

    def parse_inputs(self, text_in):
        warning_field = []
        company = text_in['entityId']
        userid = company + '-' + str(text_in['userId'])
        if 'vendor' in text_in.keys():
            if isinstance(text_in['vendor'], str):
                vendor_raw = text_in['vendor']
                vendor_txt = re.sub("([^\w]|[ 0-9_])", '', vendor_raw.lower())
                # vendor = str(userid) + '-' + vendor_txt
                vendor = company + '-' + vendor_txt
            else:
                vendor = ''
                warning_field.append('Vendor field sent in, but not a string')
        else:
            vendor = ''

        if 'receiptAmt' in text_in.keys():
            amount = text_in['receiptAmt']
            try:
                amount = float(amount)
            except:
                amount = ''
                warning_field.append('Amount field cannot be converted to float')
        else:
            amount = ''
        if amount:
            if amount == 0:
                amtbucket = '0.0'
            elif amount < 10:
                amtbucket = '<=10'
            elif amount < 50:
                amtbucket = '<=50'
            elif amount < 100:
                amtbucket = '<=100'
            elif amount < 1000:
                amtbucket = '<=1000'
            else:
                amtbucket = '>1000'
        else:
            amtbucket = ''

        return company, userid, vendor, amount, amtbucket, warning_field

    def remap(self, text_in):
        histories_found = []
        # New model from Everaldo maps receipt to DS type from ocr text with 88% accuracy
        vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
        clf = joblib.load('new_ds_model/model.pkl')
        x = vectorizer.transform([text_in['ocrText']])
        pred = clf.predict(x)[0]

        # Get the necessary information out of the input dictionary.
        company, userid, vendor, amount, amtbucket, warn_field = self.parse_inputs(text_in)

        # I definitely need the allowed types passed in as an argument or this is going to end poorly
        # This is checked at Type_Server level
        allowed_list = {x['ExpKey']: x['Name'] for x in text_in['userExpenseTypes']}

        # Get history information out of couchbase
        user_hist, comp_hist, vend_hist, amt_hist = self.get_history(company, userid, vendor, amount)

        # Create a new allowed list from the allowed types in the DS category.
        # Save the full allowed list, just in case
        new_allowed_list = {}
        for k, v in allowed_list.items():
            if v.startswith('z') or v.startswith('xx') or v.startswith('Z'):
                continue
            if DS_type_mapping.to_ds_types(k, v, self.type_order, self.type_dict) == pred:
                new_allowed_list[k] = v

        # Start off the point-giving by giving every category in the predicted DS type a big point boost.
        et_guess = Counter()
        for et_key in new_allowed_list.keys():
            et_guess[et_key] += 50

        # This is the section that deals with time in Meals.
        hfm = None
        if pred == "MEALS":
            te = time_extractor.TimeExtractor()
            minutes_from_midnight = te.extract_time(text_in['ocrText'])
            if minutes_from_midnight['time'] and minutes_from_midnight['time'][0]['value'] != -1:
                mfm = minutes_from_midnight['time'][0]['value']
                hfm = float(mfm) / 60
                if mfm < 630:  # 10:30am
                    searchstr = ['brk', 'bfast', 'break']
                elif mfm < 840:  # 2:00pm
                    searchstr = ['lun']
                elif mfm > 1020:  # 5:00pm
                    searchstr = ['din']
                else:
                    searchstr = []
                for k, v in new_allowed_list.items():
                    for s in searchstr:
                        if s in v.lower():
                            et_guess[k] += 8

        # It's always XXXXX
        if pred == 'LODNG':
            et_guess['LODNG'] += 3
        if pred == 'CARRT':
            et_guess['CARRT'] += 3
        if pred == 'TELEC':
            et_guess['CELPH'] += 1
        if pred == 'OFCSP':
            et_guess['OFCSP'] += 3
        if pred == 'GRTRN':
            et_guess['TAXIX'] += 3
        if pred == 'PARKG':
            et_guess['PARKG'] += 3

        # This is the section where you give points (or subtract points) based on history files
        if amt_hist:
            et_guess = self.cross_check(et_guess, amt_hist, new_allowed_list, 5)
            histories_found.append('amount')
        if vend_hist:
            et_guess = self.cross_check(et_guess, vend_hist, new_allowed_list, 7)
            histories_found.append('vendor')
        if user_hist:
            et_guess = self.cross_check(et_guess, user_hist, new_allowed_list, 6)
            histories_found.append('user')
        if comp_hist:
            et_guess = self.cross_check(et_guess, comp_hist, new_allowed_list, 5)
            histories_found.append('entity')

        # If DS type is wrong, it's possible to have zero results in et_guess.  This section will go
        # back and make sure there's something reasonable to guess, even if it's some other DS type
        # If all else fails, it'll put UNDEF, but so far, that's 2 out of 20,000+ cases.
        if not len(et_guess):
            if vend_hist:
                et_guess = self.cross_check(et_guess, vend_hist, allowed_list, 5)
            elif user_hist:
                et_guess = self.cross_check(et_guess, user_hist, allowed_list, 5)
            elif comp_hist:
                et_guess = self.cross_check(et_guess, comp_hist, allowed_list, 5)
        if not len(et_guess):
            et_guess['UNDEF'] += 1

        # total_counts = sum(et_guess.values())
        list_out = []
        for k, v in et_guess.most_common(5):
            list_out.append({'type': k, 'score': v})

        return_dictionary = {'expenseTypes': list_out}

        logging_dictionary = {'ETlog - predicted DS type': pred,
                              'ETlog - amount used': amount,
                              'ETlog - vendor used': vendor,
                              'ETlog - time used': hfm,
                              'ETlog - histories found': histories_found}

        if warn_field:
            logging_dictionary['!! WARNING !!'] = warn_field

        return_dictionary.update(logging_dictionary)

        return return_dictionary


def parse_inputs(p, file_type=1):
    # file_type = 0: mike's version
    # file_type = 1: everaldo's benchmark data set version
    if file_type == 0:
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
        ds_req = p[12]
    elif file_type == 1:
        date_string = p[0]
        entity = p[1]
        imageid = p[2]
        ds_req = p[3]
        datekey = p[4]
        amount = p[5]
        currency = p[6]
        location_city = p[7]
        location_state = p[8]
        location_country = p[9]
        vendor = p[10]
        exp_name = p[11]
        exp_key = p[12].replace('\n', '')
    return entity, exp_name, exp_key, ds_req, imageid


def process_all(file_name, file_type=1):
    total_file_count = 0
    no_ds_request = 0
    ds_correct = 0
    et_correct = 0
    amount_used = 0
    user_used = 0
    entity_used = 0
    vendor_used = 0
    time_found = 0
    company_dict = {}
    company_types = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            total_file_count += 1
            if total_file_count == 1:
                continue
            # if total_file_count > 1000:
            #     break
            p = line.split('\t')
            entity, exp_name, exp_key, ds_req, imageid = parse_inputs(p, file_type)

            if len(ds_req) < 2:
                no_ds_request += 1
                # print('no ds request')
                continue
            data_input = json.loads(ds_req)

            output = remap_class.remap(data_input)

            if entity not in company_dict.keys():
                company_dict[entity] = Counter()
                company_types[entity] = {'num types': len(data_input['userExpenseTypes']),
                                         'num meals': len([z for z in data_input['userExpenseTypes']
                                                        if DS_type_mapping.to_ds_types(z['ExpKey'], z['Name'],
                                                        remap_class.type_order, remap_class.type_dict) == 'MEALS'])}
            company_dict[entity]['total'] += 1

            # print(output['expenseTypes'][0]['type'], exp_key)

            if output['expenseTypes'][0]['type'] == exp_key:
                et_correct += 1
                company_dict[entity]['et correct'] += 1
            real_ds_type = DS_type_mapping.to_ds_types(exp_key, exp_name, remap_class.type_order, remap_class.type_dict)
            if real_ds_type == output['ETlog - predicted DS type']:
                ds_correct += 1
                company_dict[entity]['ds correct'] += 1
            if 'amount' in output['ETlog - histories found']:
                amount_used += 1
            if 'vendor' in output['ETlog - histories found']:
                vendor_used += 1
            if 'entity' in output['ETlog - histories found']:
                entity_used += 1
            if 'user' in output['ETlog - histories found']:
                user_used += 1
            if output['ETlog - time used']:
                time_found += 1

    print("total file number: " + str(total_file_count))
    print("no ds request: " + str(no_ds_request))
    print("total processed: " + str(total_file_count - no_ds_request))
    print("number DS correct: " + str(ds_correct))
    print("number ET correct: " + str(et_correct))
    print("amount found: " + str(amount_used))
    print("vendor found: " + str(vendor_used))
    print("entity found: " + str(entity_used))
    print("user history found: " + str(user_used))
    print("time found: " + str(time_found))

    outfile = open('JanuaryTypes.txt', 'w+', encoding='utf-8')
    outfile.write("Company\tDS Correct\tET Correct\tCount\tNum Types\tNum MealTypes\n")
    for c in company_dict.keys():
        outfile.write(c + '\t')
        outfile.write(str(company_dict[c]['ds correct']) + '\t')
        outfile.write(str(company_dict[c]['et correct']) + '\t')
        outfile.write(str(company_dict[c]['total']) + '\t')
        outfile.write(str(company_types[c]['num types']) + '\t')
        outfile.write(str(company_types[c]['num meals']) + '\n')


def process_some(file_name, file_type=1):
    file_root = 'bm_errors_'
    total_file_count = 0
    no_ds_request = 0
    ds_correct = 0
    et_correct = 0
    amount_used = 0
    user_used = 0
    entity_used = 0
    vendor_used = 0
    time_found = 0
    error_count = 0

    ds_api_url = 'http://seapr1dsweb.concurasp.com:80/ds-webapi/service/expenseClassification/receiptTypeClassification'
    request_type = {'Content-Type': 'application/json'}

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            total_file_count += 1
            # if total_file_count > 1000:
            #     break
            p = line.split('\t')
            entity, exp_name, exp_key, ds_req, imageid = parse_inputs(p, file_type)
            if len(ds_req) < 2:
                no_ds_request += 1
                continue
            if file_type and total_file_count == 1:
                continue
            data_input = json.loads(ds_req)

            output = remap_class.remap(data_input)

            data = json.dumps(data_input).encode('utf-8', 'ignore')
            req = urllib.request.Request(ds_api_url, data, request_type)
            fangs_output = urllib.request.urlopen(req)
            t_str = fangs_output.read().decode('utf-8')
            t_two = json.loads(t_str)
            fangs_pred = t_two['expenseTypes'][0]['type'] if len(t_two['expenseTypes']) else ''
            print(fangs_pred)

            real_ds_type = DS_type_mapping.to_ds_types(exp_key, exp_name, remap_class.type_order, remap_class.type_dict)
            if real_ds_type == output['ETlog - predicted DS type']:
                ds_correct += 1
            if 'amount' in output['ETlog - histories found']:
                amount_used += 1
            if 'vendor' in output['ETlog - histories found']:
                vendor_used += 1
            if 'entity' in output['ETlog - histories found']:
                entity_used += 1
            if 'user' in output['ETlog - histories found']:
                user_used += 1
            if output['ETlog - time used']:
                time_found += 1
            if output['expenseTypes'][0]['type'] == exp_key:
                et_correct += 1
            else:
                new_text = str(data_input['ocrText']).replace('\t', '').replace('\r\n', '')
                new_types = {x['ExpKey']: x['Name'] for x in data_input['userExpenseTypes']}
                pred_types = [o['type'] for o in output['expenseTypes']]
                new_preds = []
                for p in pred_types:
                    if p in new_types.keys():
                        new_preds.append(str(p)+': '+str(new_types[p]))
                real_name = new_types[exp_key] if exp_key in new_types.keys() else ''
                if error_count % 100 == 0:
                    outfile_num = '{:02d}'.format(int(error_count / 100))
                    outfile = open(file_root + outfile_num + '.txt', 'w+', encoding='utf-8')
                    outfile.write('imageID\tocrText\ttime found\tuserExpenseTypes\t')
                    outfile.write('PredDS\tFangsPred\tRealET\tRealET Name\tPredET\tError Type\n')
                outfile.write(str(imageid) + '\t' + new_text + '\t')
                outfile.write(str(output['ETlog - time used']) + '\t' + str(new_types) + '\t')
                outfile.write(str(output['ETlog - predicted DS type']) + '\t')
                outfile.write(fangs_pred + '\t' + exp_key + '\t' + real_name + '\t' + str(new_preds) + '\t')
                if real_ds_type != output['ETlog - predicted DS type']:
                    outfile.write('DS TYPE WRONG\n')
                else:
                    if exp_key in pred_types:
                        outfile.write('RANK ERROR\n')
                    else:
                        outfile.write('TYPE 3 ERROR\n')
                error_count += 1
            if error_count > 1100:
                break

    print("total file number: " + str(total_file_count))
    print("no ds request: " + str(no_ds_request))
    print("total processed: " + str(total_file_count - no_ds_request))
    print("number DS correct: " + str(ds_correct))
    print("number ET correct: " + str(et_correct))
    print("amount found: " + str(amount_used))
    print("vendor found: " + str(vendor_used))
    print("entity found: " + str(entity_used))
    print("user history found: " + str(user_used))
    print("time found: " + str(time_found))


def single_instance():
    # Single Instance Test
    test = {"entityId": "p00425z4gu", "userId": "27341",
            "receiptAmt": "12.50",
            "ocrText": "this is a dinner receipt. at yoshi's. thanks",
            "userExpenseTypes": [{"ExpKey": "01001", "Name": "Airline Fees", "SpdCat": "AIRFR"},
                                 {"ExpKey": "01003", "Name": "Employee Events", "SpdCat": "OTHER"},
                                 {"ExpKey": "01007", "Name": "Voith Only - Employee Group Meals", "SpdCat": "ENTER"},
                                 {"ExpKey": "01020", "Name": "Ferry Transportation", "SpdCat": "OTHER"},
                                 {"ExpKey": "01023", "Name": "Coffee, water, beverage", "SpdCat": "MEALS"},
                                 {"ExpKey": "AIRFR", "Name": "Airfare", "SpdCat": "AIRFR"},
                                 {"ExpKey": "BRKFT", "Name": "Breakfast", "SpdCat": "MEALS"},
                                 {"ExpKey": "BUSML", "Name": "Business Meal(attendees)", "SpdCat": "ENTER"},
                                 {"ExpKey": "CARRT", "Name": "Car Rental", "SpdCat": "CARRT"},
                                 {"ExpKey": "CELPH", "Name": "Cellular Phone", "SpdCat": "TELEC"},
                                 {"ExpKey": "DINNR", "Name": "Dinner", "SpdCat": "MEALS"},
                                 {"ExpKey": "GASXX", "Name": "Gasoline", "SpdCat": "GASXX"},
                                 {"ExpKey": "LODNG", "Name": "Hotel", "SpdCat": "LODGA"},
                                 {"ExpKey": "LUNCH", "Name": "Lunch", "SpdCat": "MEALS"},
                                 {"ExpKey": "OFCSP", "Name": "Office Supplies", "SpdCat": "OFFIC"},
                                 {"ExpKey": "ONLIN", "Name": "Internet", "SpdCat": "TELEC"},
                                 {"ExpKey": "PARKG", "Name": "Parking", "SpdCat": "GRTRN"},
                                 {"ExpKey": "TAXIX", "Name": "Taxi", "SpdCat": "GRTRN"},
                                 {"ExpKey": "UNDEF", "Name": "Undefined", "SpdCat": "OTHER"}]
            }
    remap_class = Remapper()
    output = remap_class.remap(test)
    print(output)

if __name__ == '__main__':
    # January expenses for graphing
    remap_class = Remapper()

    jan_file = 'C:/Users/audreyc/PyCharm/ECTokenEval/token-test-dsreq_2016-07-07_1534'
    benchmark_file = 'C:/Users/audreyc/PyCharm/ECTokenEval/expenseit_token_benchmark_dataset.tsv'
    # process_some(jan_file, 0)
    # process_some(benchmark_file, 1)
    process_all(jan_file, 0)
    # process_all(benchmark_file, 1)

    # single_instance()
