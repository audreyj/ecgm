import json
import pickle
from collections import Counter


def parse_input():
    file_dict = {'entityhistory': 'new_ds_model/allexpense_entityhistory.tsv'}
        # {'vendorhist': 'new_ds_model/expense_entityvendorhistory.tsv',
        #          'companyhist': 'new_ds_model/expense_entityhistory.tsv',
        #          'userhist': 'new_ds_model/expense_userhistory.tsv'}
    for output_name, file_name in file_dict.items():
        with open(file_name, 'r', encoding='utf-8') as f:
            output_dict = {}
            for line in f:
                p = line.split('\t')
                raw_entity_key = p[0]
                sub_df = json.loads(p[1])
                entity_key = raw_entity_key.split(']')[1]
                output_dict[entity_key] = Counter()
                for exp_key, count_val in sub_df.items():
                    output_dict[entity_key][exp_key] += count_val

        pickle.dump(output_dict, open('new_ds_model/allexpense_' + output_name + '.pkl', 'wb'))


if __name__ == '__main__':
    parse_input()

