"""
author: audreyc
Open a dataset.  Sums what DS-types are in every entry
"""

import pickle
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import DS_type_mapping

bad_chars = {'\\r': ' ', '\\n': ' ', '•': ' ',
             '1': '•', '2': '•', '3': '•', '4': '•', '5': '•',
             '6': '•', '7': '•', '8': '•', '9': '•', '0': '•'}

spend_cats = ['MEALS', 'GRTRN', 'LODGA', 'AIRFR',  'CARRT', 'GASXX',
              'OFFIC', 'OTHER', 'TELEC', 'SHIPG', 'ENTER', 'FEESD', 'TRAIN']


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
    line_count = 0
    for line in contents:
        line_count += 1
        if max_lines and line_count > max_lines:
            break
        t = line.split(sep)
        if len(t) != len(column_names):
            print("problem with column numbers, line " + str(line_count))
            break
        for e, c in enumerate(column_names):
            if c == 'datekey':
                out_dict[c][int(t[e])] += 1
            else:
                out_dict[c][t[e]] += 1
        # if t[1] == '27341':
        #     print(line)
    print("%d lines read from file" % line_count)
    for k, v in out_dict.items():
        if k == 'ocr':
            continue
        print(k, end=': ')
        print(v.most_common(20))
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
    type_order, type_dict = DS_type_mapping.query_file()
    ds_type_counter = Counter()
    for exp_name, num_counts in data_dict['expense_name'].most_common():
        this_ds_type = DS_type_mapping.to_ds_types(exp_name, type_order, type_dict)
        ds_type_counter[this_ds_type] += num_counts
    print(ds_type_counter)

if __name__ == "__main__":
    run_all(sys.argv[1])

