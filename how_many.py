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

file_name = 'error_analysis/exp-feb'

expense_counter = Counter()
expenseit_counter = Counter()
type_order, type_dict = DS_type_mapping.query_file()

line_count = 0
max_lines = 0
with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        if max_lines and line_count > max_lines:
            break
        if line_count % 100000 == 0:
            print(line_count)
        p = line.split('\t')
        entity = p[0]
        user = p[1]
        datekey = p[2]
        expense_key = p[3]
        expense_name = p[4]
        amount = p[5]
        vendor = p[6]
        expenseit_flag = int(p[7])

        ds_type = DS_type_mapping.to_ds_types(expense_name, type_order, type_dict)

        expense_counter[ds_type] += 1

        if expenseit_flag:
            expenseit_counter[ds_type] += 1

print(sum(expense_counter.values()))
print(expense_counter)

print('-------------')
print(sum(expenseit_counter.values()))
print(expenseit_counter)