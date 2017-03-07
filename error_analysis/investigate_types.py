import pickle
import re
import os
from collections import Counter

types_dict = pickle.load(open('entity_expense_types.pkl', 'rb'))


def parse_client_file():
    client_file = '../../company_segments/client_salesforce.csv'
    total_line_count = 0
    no_smb_cat = 0
    smb_dict = {}
    with open(client_file, 'r', encoding='utf-16') as f:
        for line in f:
            total_line_count += 1
            if total_line_count == 1:
                continue
            p = line.split('\t')
            entity_name = p[0]
            entity_size = p[1]
            entity_id = p[2]
            smb_cat = p[6]
            vertical = p[-1].replace('\n', '')
            if len(smb_cat) < 2:
                no_smb_cat += 1
                continue

            if entity_id not in smb_dict.keys():
                smb_dict[entity_id] = {'Name': entity_name, 'Size': entity_size,
                                       'Category': smb_cat, 'Vertical': vertical}

    print("total lines assessed: " + str(total_line_count))
    print("no smb category: " + str(no_smb_cat))
    pickle.dump(smb_dict, open('smb_dict_new.pkl', 'wb'))


if not os.path.isfile('smb_dict_new.pkl'):
    parse_client_file()

smb_dict = pickle.load(open('smb_dict_new.pkl', 'rb'))
total_type_number = 0
entity_missing = 0
output_dict = {}
for entity, types in types_dict.items():
    total_type_number += len(types)
    for t in types:
        if entity in smb_dict.keys() and smb_dict[entity] != '-':
            vert = smb_dict[entity]['Vertical']
            if vert not in output_dict.keys():
                output_dict[vert] = Counter()
            step_one = re.sub(u'[^\w]+', ' ', t['Name'])
            out_key = step_one.lower().capitalize()
            output_dict[vert][out_key] += 1
        else:
            entity_missing += 1

print(total_type_number)
print("Missing entities from smb file: ", str(entity_missing))
print(len(output_dict))
for k, v in output_dict.items():
    print(k)
    if k == '#N/A':
        this_file = open('None.txt', 'w+')
    else:
        this_file = open(k + '.txt', 'w+')
    for o in v.most_common(100):
        # print(o)
        this_file.write(o[0] + ', ' + str(o[1]) + '\n')
    # input('--- pause ---')
    this_file.close()
