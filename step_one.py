"""
author: audreyc
Last Update: 04/27/16
The first step is to parse the train and test datafiles (separately) so they're in a state ready for modeling.
Also, during this step, I remove the //r//n and extra spaces from the OCR text.
Input:
    - file name (of file probably stored in inputdata)
    - 0 to indicated training data or 1 to indicate test data
        (ultimately they're treated the same, just different names on output files)
    - use_ds_types 0 (off) or 1 (on)
        (will change the target list to being in ds types, expense types otherwise)
    - overwrite_history 0 (off) or 1 (on)
        (overwrites user and company history)
    - expense_type_names 0 (off) or 1 (on)
        (instead of ds type key or expense type key, write expense name directly to target list)
Output (all to intermediate folder):
    - [entity_text]_x_[subtext][train|test][key_type].pkl  (e.g. "concur_x_trainDS.pkl")
        - entity_text: either 'concur' if the word 'concur' appears in the input file name or 'all'
        - x: represents the document list (almost always the semi-cleaned OCR text)
        - subtext: anything that's in the input file name that's not "inputdata/" or "concur_" or "set"
            (it's descriptive)
        - train|test: input will tell code whether this is a trainset or a testset.
            (other versions of this file parsing will cross validate inside the code)
        - key_type: whether the target list is DS type or ET (expense type)
            (actually has no bearing on the x files since they are all OCR input text)
    - [entity_text]_y_[subtext][train|test][key_type].pkl
        - y: represents the target list (the 'correct' answers)
    - [entity_text]_UserHist[key_type].pkl and [entity_text]_Hist[key_type].pkl
        - only if overwrite_history is on, the code will also write out (overwrite) history files
        - each pickle is a dictionary
            - user-key (entity-userid): Counter of [key_types]
            - entity : Counter of [key_types]
    - info_[subtext][key_type].pkl
         - only if test data
         - a list in the same order as the test document and target lists, with more info on each receipt item
"""

bad_chars = {'\\r': ' ', '\\n': ' ', '•': ' ',
             '1': '•', '2': '•', '3': '•', '4': '•', '5': '•',
             '6': '•', '7': '•', '8': '•', '9': '•', '0': '•'}

import sys
import DS_type_mapping as type_map
import pickle
from collections import Counter

overwrite_history = 0
expense_type_names = 0

savedir = 'intermediate/'


def parse_file(file_name, test_train, use_ds_types):
    subtext = file_name.replace('inputdata/', '').replace('concur_', '').replace('set', '')
    entity_text = 'concur' if 'concur' in file_name else 'all'
    type_order, type_dict = type_map.query_file()
    Tsuffix = '_'+subtext+'test' if test_train else '_'+subtext
    DSsuffix = 'DS' if use_ds_types else 'ET'
    if expense_type_names:
        DSsuffix = DSsuffix+'N'
    suffix = Tsuffix + DSsuffix

    company_x = []
    company_y = []
    user_dict = {}
    info_list = []
    line_counter = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line_counter += 1
            p = line.split('\t')
            entity = p[0]
            userid = p[1]
            datekey = int(p[2])
            expense_key = p[3]
            expense_name = p[4]
            longer_type = True if len(p) > 6 else False
            # if entity != 'p00425z4gu':
            #     continue
            if not longer_type:
                ocr = p[5]
            else:
                amount = p[5]
                vendor = p[6]
                expenseit = p[7]
                ocr = p[8]

            if use_ds_types:
                expense_key = type_map.to_ds_types(expense_key, expense_name, type_order, type_dict)

            t = ocr
            for bc, rw in bad_chars.items():
                t = t.replace(bc, rw)
            t = t.lower()
            s = t.split()
            s = [x for x in s if len(x) > 1]
            ocr = ' '.join(s)

            company_x.append(ocr)
            if expense_type_names:
                company_y.append(expense_name)
            else:
                company_y.append(expense_key)

            if line_counter == 2:
                print("entity: %s, userid: %s, exp_name: %s, ds_type: %s" % (entity, userid, expense_name, expense_key))
                # print("ocr: " + ocr)

            if test_train == 0:  # if this is the train set, create UserHist file
                if userid in user_dict.keys():
                    user_dict[userid][expense_key] += 1
                else:
                    user_dict[userid] = Counter()
                    user_dict[userid][expense_key] += 1
            else:
                info_list.append({'datekey': datekey, 'entity': entity, 'userid': userid})
                if longer_type:
                    info_list[-1].update({'amount': amount, 'vendor': vendor, 'expenseit': expenseit})

    pickle.dump(company_x, open(savedir + entity_text + '_x' + suffix + '.pkl', 'wb'))
    pickle.dump(company_y, open(savedir + entity_text + '_y' + suffix + '.pkl', 'wb'))
    if test_train == 0:  # if this is a train set, create UserHist and CompanyHist files
        if overwrite_history:
            company_dict = {}
            temp_counter = Counter()
            for k, v in user_dict.items():
                temp_counter += v
            company_dict[entity] = temp_counter
            print(company_dict[entity].most_common())
            pickle.dump(company_dict, open(savedir + entity_text + '_hist' + DSsuffix + '.pkl', 'wb'))
            pickle.dump(user_dict, open(savedir + entity + '_UserHist' + DSsuffix + '.pkl', 'wb'))
            print(user_dict['27341'])
    else:
        pickle.dump(info_list, open(savedir + "info_" + subtext + DSsuffix + ".pkl", 'wb'))

    print("%d entries saved to: %s" % (len(company_x), savedir+entity_text+'_x'+suffix+'.pkl'))
    print("info_list length: ", len(info_list))

if __name__ == "__main__":
    # parse_file(sys.argv[1], sys.argv[2], sys.argv[3])
    parse_file('inputdata/concur_trainset2', 1, 1)  # filename, train=0 | test=1, ds_types on=1 | off=0
