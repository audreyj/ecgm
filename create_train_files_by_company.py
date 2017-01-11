"""
author: audreyc
Last Update: 04/28/16
The first step is to parse the train and test datafiles so they're in a state ready for modeling.
This particular file will also cross validate the trainset into train and test.
Also, during this step, I remove the //r//n and extra spaces from the OCR text.
Input:
    - file name (of file probably stored in inputdata)
    - use_ds_types 0 (off) or 1 (on)
        (will change the target list to being in ds types, expense types otherwise)
    - cross_validate 0 (off) or 1 (on)
        (if on, separates into test and train files.  If off, only creates train files)
Output (all to intermediate folder):
    - [entity_text]_x_[subtext][train|test].pkl  (e.g. "concur_x_trainDS.pkl")
        - entity_text: either 'concur' if the word 'concur' appears in the input file name or 'all'
        - x: represents the document list (almost always the semi-cleaned OCR text)
        - subtext: anything that's in the input file name that's not "inputdata/" or "concur_" or "set"
            (it's descriptive)
        - train|test: input will tell code whether this is a trainset or a testset.
            (other versions of this file parsing will cross validate inside the code)
    - [entity_text]_y_[subtext][train|test][key_type].pkl
        - y: represents the target list (the 'correct' answers)
        - key_type: whether the target list is DS type or ET (expense type)
    - info_[subtext].pkl
         - a list in the same order as the document and target lists, with more info on each receipt item
"""

bad_chars = {'\\r': ' ', '\\n': ' ', '•': ' ',
             '1': '•', '2': '•', '3': '•', '4': '•', '5': '•',
             '6': '•', '7': '•', '8': '•', '9': '•', '0': '•'}

import sys
import DS_type_mapping as type_map
import pickle
from collections import Counter

savedir = 'intermediate/'


def parse_file(company, cross_validate=True):
    file_name = 'inputdata/' + company + '_trainset'
    type_order, type_dict = type_map.query_file()
    doc_list = []
    et_target_list = []
    ds_target_list = []
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
            if not longer_type:
                ocr = p[5]
            else:
                amount = p[5]
                vendor = p[6]
                expenseit = p[7]
                ocr = p[8]

            ds_key = type_map.to_ds_types(expense_name, type_order, type_dict)

            t = ocr
            for bc, rw in bad_chars.items():
                t = t.replace(bc, rw)
            t = t.lower()
            s = t.split()
            s = [x for x in s if len(x) > 1]
            ocr = ' '.join(s)

            doc_list.append(ocr)
            et_target_list.append(expense_key)
            ds_target_list.append(ds_key)

            # if line_counter == 2:
            #     print("entity: %s, userid: %s, exp_name: %s, et_type: %s, ds_type: %s" %
            #           (entity, userid, expense_name, expense_key, ds_key))
                # print("ocr: " + ocr)

            info_list.append({'datekey': datekey, 'entity': entity, 'userid': userid})
            if longer_type:
                info_list[-1].update({'amount': amount, 'vendor': vendor, 'expenseit': expenseit})

    if cross_validate:
        # x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        #                                           doc_list, target_list, test_size=0.1, random_state=1)

        ninety = int(len(doc_list) * 0.9)
        x_train = doc_list[:ninety]
        x_test = doc_list[ninety:]
        y_train_et = et_target_list[:ninety]
        y_train_ds = ds_target_list[:ninety]
        y_test_et = et_target_list[ninety:]
        y_test_ds = ds_target_list[ninety:]
        info_train = info_list[:ninety]
        info_test = info_list[ninety:]

        pickle.dump(x_test, open(savedir + company + '_x_test.pkl', 'wb'))
        pickle.dump(y_test_et, open(savedir + company + '_y_testET.pkl', 'wb'))
        pickle.dump(y_test_ds, open(savedir + company + '_y_testDS.pkl', 'wb'))
        pickle.dump(info_test, open(savedir + company + '_info_test.pkl', 'wb'))
    else:
        x_train = doc_list
        y_train_et = et_target_list
        y_train_ds = ds_target_list
        info_train = info_list

    pickle.dump(x_train, open(savedir + company + '_x_train.pkl', 'wb'))
    pickle.dump(y_train_et, open(savedir + company + '_y_trainET.pkl', 'wb'))
    pickle.dump(y_train_ds, open(savedir + company + '_y_trainDS.pkl', 'wb'))
    pickle.dump(info_train, open(savedir + company + '_info_train.pkl', 'wb'))

if __name__ == "__main__":
    parse_file(sys.argv[1])
    # parse_file('inputdata/emc_trainset')  # filename, ds_types on=1 | off=0
