"""
author: audreyc
Investigate a specific dataset from the OCR database,
filter for a specific person or two,
assess their individual errors,
run through v3
"""

import pickle
import json
import sys
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from collections import Counter

bad_chars = {'\\r': ' ', '\\n': ' ', '•': ' ',
             '1': '•', '2': '•', '3': '•', '4': '•', '5': '•',
             '6': '•', '7': '•', '8': '•', '9': '•', '0': '•'}

spend_cats = ['MEALS', 'GRTRN', 'LODGA', 'AIRFR',  'CARRT', 'GASXX',
              'OFFIC', 'OTHER', 'TELEC', 'SHIPG', 'ENTER', 'FEESD', 'TRAIN']

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


def open_file(file_name):
    """
    this function will take any file name and try to recognize it and open it
    - pickle files: hopefully load a dict, that'll have names built in, so skip the column name search
    - json files: hopefully load a dict, that'll have names built in, so skip the column name search
    - txt or no extension: should be a tab delimited file, search the file for column names and then read line by line
    """
    print("Opening File Name: %s" % file_name)
    if file_name.endswith('.pkl'):
        f = pickle.load(open(file_name, 'rb'))
        sep = "skip" if isinstance(f, dict) else "list"
        if sep == "list":
            print("Pickle is list.  First 2 entries:")
            print(f[0])
            print(f[1])
            print("Length of list: ", len(f))
    elif file_name.endswith('.json'):
        f = json.loads(open(file_name, 'rb'))
        sep = "skip"
    else:
        f = open(file_name, 'r', encoding='utf-8')
        sep = "\t"
    return f, sep


def check_header(first_line, sep):
    """
    this checks the first line of the file (given) and if most of these are fully alphabet chars, then assume
    it's a given header row and load those as names, and continue to the rest of the file.
    otherwise, match item by item against a bunch of if statements to the expected values
    """
    entries = first_line.split(sep)
    count_alpha = 0
    unknowns = 0
    for t in first_line:  # iterate over string
        if t.isalpha():
            count_alpha += 1
    if float(count_alpha) / len(first_line) > 0.9:
        print("Appears to have a header row: %s" % entries)
        return entries, 0
    c = []
    for t in entries:
        if t.startswith('p') and 'entityID' not in c:
            c.append('entityID')
        elif t.startswith('20') and len(t) == 8 and 'datekey' not in c:
            c.append('datekey')
        elif t in spend_cats and 'spendcat' not in c:
            c.append('spendcat')
        elif t.startswith('0') and len(t) == 5 and 'expense_key' not in c:
            # This assumes expense_key will be the FIRST five digit code...
            c.append('expense_key')
        elif t.isalpha() and len(t) == 5 and 'expense_key' not in c:
            c.append('expense_key')
        elif '\\r\\n' in t and 'ocr' not in c:
            c.append('ocr')
        elif t.isdigit() and 'userID' not in c:
            c.append('userID')
        elif t.split('.')[-1].isdigit() and 'amount' not in c:
            c.append('amount')
        elif t.split() and t.split()[0].isalpha() and 'expense_name' not in c:
            c.append('expense_name')
        else:
            c.append('unknown' + str(unknowns))
            unknowns += 1
    print(first_line)
    print("Guessing column names: %s" % c)
    return c, 1


def get_data(contents, column_names, max_lines, sep):
    """
    this cycles through the rest of the file (up to max_lines) and counts instances of everything into a dict
    """
    out_dict = {col: Counter() for col in column_names}
    entity_num = column_names.index('entityID')
    user_num = column_names.index('userID')
    ocr_num = column_names.index('ocr')
    user_search = ['28177', '611', '30540']
    line_count = 0
    non_skipped = 0
    for line in contents:
        line_count += 1
        if max_lines and line_count > max_lines:
            break
        t = line.split(sep)
        if len(t) != len(column_names):
            print("problem with column numbers, line " + str(line_count))
            break
        if t[entity_num] != 'p00425z4gu':
            continue
        if t[user_num] not in user_search:
            continue
        non_skipped += 1
        for e, c in enumerate(column_names):
            if c == 'datekey':
                out_dict[c][int(t[e])] += 1
            else:
                out_dict[c][t[e]] += 1
        print(line)
    print("total lines: %d" % line_count)
    print("not skipped: %d" % non_skipped)
    return out_dict


def run_all(file_name, max_lines=0):
    """
    this runs each piece in the correct order, with just the filename and max_lines as inputs
    I broke into these pieces because I might be able to use each piece individually
    """
    file_contents, sep = open_file(file_name)
    if sep != 'skip':
        columns, skip_to_next = check_header(file_contents.readline(), sep)
        if not skip_to_next:
            file_contents.seek(0)
        data_dict = get_data(file_contents, columns, max_lines, sep)
    else:
        data_dict = file_contents
        columns = file_contents.keys()


if __name__ == "__main__":
    # run_all(sys.argv[1])
    # run_all('sampledata2/sampled_Y_trainDS.pkl')
    run_all('C:/Users/audreyc/PyCharm/octset2')
