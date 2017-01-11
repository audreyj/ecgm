"""
author: audreyc
Re-write of DS type_mapping functions
"""

import pickle
from collections import Counter


def query_file():
    type_dict = {}
    type_order = []
    with open('inputdata/ExpenseTypeMapping.tsv', 'r') as file:
        for line in file:
            t = line.split('\t')
            type_dict[t[0]] = [s.replace('\n', '').lower() for s in t]
            type_order.append(t[0])
    # print(type_dict)
    return type_order, type_dict


def to_ds_types(type_name, type_order=None, type_dict=None):
    if not isinstance(type_name, str):
        type_name = str(type_name)
    new_type_name = type_name.replace(' - ', ' ').lower()
    if not type_order:  # This ought to make it faster when doing many in a row
        type_order, type_dict = query_file()
    for t in type_order:
        for each_phrase in type_dict[t]:
            if each_phrase in new_type_name:
                return t
    return 'OTHER'


if __name__ == "__main__":
    print(to_ds_types("Car Rental"))
    print(to_ds_types("hotel test"))
    print(to_ds_types("employee-only entertainment"))
