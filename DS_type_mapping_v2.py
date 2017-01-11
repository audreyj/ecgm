"""
author: audreyc
A new version of the DS type mapper:
 - all contained in one file
 - makes two passes through words
"""

import re
from collections import Counter
import pandas as pd


token_order = ['MEALS', 'CARRT', 'GRTRN', 'LODNG', 'AIRFR', 'GASXX', 'PARKG', 'OFCSP', 'TELEC',
               'TRANG', 'FIELD', 'OTHER']

comb_any = {'LODNG': ['hotel', 'lodging', 'accomod', 'accommod'],
            'MEALS': ['meal', 'm&', 'entertain', 'subsist', 'drink', 'refresh', '&m', '&e',
                      'breakf', 'brk', 'bfast', 'lun', 'dinn', 'snack', 'beverage', 'cater',
                      'diem', 'allowance', 'incident', 'food', 'drink', 'dining', 'banquet',
                      'coffee', 'water', 'donut', 'pantry',  'cafe', 'custent', 'staffent',
                      'grocer', 'sustenance'],
            'AIRFR': ['airfare', 'airline', 'flight', 'baggage', 'luggage'],
            'GRTRN': ['taxi', 'uber', 'lyft', 'mileage', 'subway', 'ferries', 'transit', 'sedan',
                      'ground', 'commut', 'shuttle', 'limo', 'ferry', 'vehicle', 'transp', 't&'],
            'TELEC': ['phone', 'cellular', 'telec', 'fax', 'internet', 'online',
                      'data', 'at&t', 'communication'],
            'TRANG': ['dues', 'development', 'seminar', 'conference', 'education', 'tuition',
                      'course', 'professional dev', 'curric', 'subscription', 'training'],
            'OFCSP': ['suppl', 'print', 'photo', 'post', 'tools', 'copy', 'equip', 'furniture',
                      'shipp', 'office', 'stationery', 'freight', 'computer', 'software', 'hardware',
                      'storage', 'utilit', ],
            'FIELD': ['meeting', 'field', 'opening', 'convention', 'samples', 'activit', 'market',
                      'advert', 'promotion'],
            'PARKG': ['parking', 'road'],
            'CARRT': ['car rent', 'car rental', 'rental car', 'auto rental', 'vehicle rent'],
            'GASXX': ['fuel', 'gasoline', 'petrol', 'diesel', 'vehicle expense'],
            'OTHER': []}

comb_only = {'LODNG': ['room'],
             'MEALS': [],
             'AIRFR': ['air'],
             'GRTRN': ['train', 'bus', 'car', 'auto', 'public', 'fare',
                       'fares', 'rail', 'cab', 'metro', 'auto'],
             'TELEC': ['cell', 'mobile'],
             'TRANG': [], 'OFCSP': [], 'FIELD': ['event', 'events', 'show', 'shows'],
             'CARRT': [], 'PARKG': ['park', 'toll', 'tolls'],
             'GASXX': ['gas', 'oil'], 'OTHER': []}


def to_ds_types(phrase):
    for token in token_order:
        for x in comb_any[token]:
            if x in phrase.lower():
                return token
        for t in comb_only[token]:
            if re.compile(r'\b({0})\b'.format(t), flags=re.IGNORECASE).search(phrase):
                return token
    if 'travel' in phrase.lower():
        return 'GRTRN'
    return 'OTHER'

if __name__ == "__main__":
    df = pd.read_csv('new_ds_model/expenseit_token_benchmark_dataset.tsv', sep='\t')
    ds_dict = {z: Counter() for z in token_order}
    output_file = open('output_file.txt', 'w')
    for i in df.index:
        et_name = df.ix[i, 'expense_type_name']
        ds_guess = to_ds_types(et_name)
        ds_dict[ds_guess][et_name] += 1
    for s in token_order:
        output_file.write(str(s) + ' ' + str(sum(ds_dict[s].values())) + '\n')
        for each_name in ds_dict[s].most_common(50):
            output_file.write(' - -' + str(each_name) + '\n')
