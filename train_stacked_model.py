"""
author: audreyc
Last Update: 9/1/2016
This code creates training files and also trains a stacked model for expense type
training data: january ds-request data (from mike)
test data: benchmark data (from everaldo)
"""

import time
import re
import DS_type_mapping
import pickle
from collections import Counter
from sklearn import linear_model


class StackTrainer:
    def __init__(self):
        self.num_slots = 10
        self.extra_slots = 10
        self.extra_slot_labels = []
        self.x_train = []
        self.y_train = []

    def make_vector(self, existing_list, sector, list_labels, history_counter):
        top_ten = history_counter.most_common(self.num_slots)
        if len(top_ten) and len(top_ten[0][0]) > 5:
            # this catches automatically the case where the keys are '00210|blahblah'
            top_ten_labels = [v[0].split('|')[0] for v in top_ten]
        else:
            # most things have just five-digit keys
            top_ten_labels = [g[0] for g in top_ten]
        top_ten_values = [k[1] for k in top_ten]
        total_counts = sum(top_ten_values)
        for c in range(self.num_slots):
            if len(top_ten) > c and list_labels[c] in top_ten_labels:
                this_count = top_ten_values[top_ten_labels.index(list_labels[c])]
                existing_list[sector*self.num_slots+c] = (float(this_count) / float(total_counts))
        # print("vector list 0: ", existing_list[:39])
        # print("vector list 1: ", existing_list[40:79])
        # print("vector list 2: ", existing_list[80:119])
        # print("vector list 3: ", existing_list[120:159])
        # print("vector list 4: ", existing_list[160:])
        # input("pause")
        if sector != 2:
            return existing_list
        # This should be the index of the end of the regular list
        # plus 5 slots each for leftover
        this_section = 4*self.num_slots
        slot_counter = 0
        for index_history, d in enumerate(top_ten):
            if top_ten_labels[index_history] not in list_labels:
                existing_list[this_section+slot_counter] = float(d[1]) / total_counts
                self.extra_slot_labels.append(top_ten_labels[index_history])
                slot_counter += 1
                if slot_counter > self.extra_slots-1:
                    break
        return existing_list

    def create_train_files(self):
        print("---- Training Stacked Model with jan data -----")

        jan_file = 'token-test-dsreq_2016-07-07_1534'
        user_hist = pickle.load(open('new_ds_model/all_userhist.pkl', 'rb'))
        comp_hist = pickle.load(open('new_ds_model/all_companyhist.pkl', 'rb'))
        vend_hist = pickle.load(open('new_ds_model/all_vendorhist.pkl', 'rb'))
        amt_hist = pickle.load(open('inputdata/all_amounthist.pkl', 'rb'))
        # type_order, type_dict = DS_type_mapping.query_file()

        self.y_train = []
        self.x_train = []

        too_rare = 0
        no_entity_history = 0
        in_extra_slots = 0
        succeeded = 0
        no_ds_request = 0
        with open(jan_file, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.split('\t')
                entity = p[0]
                userid_raw = p[1]
                userid = entity + '-' + userid_raw
                amount = p[5]
                vendor_raw = p[9]
                if vendor_raw:
                    vendor_txt = re.sub("([^\w]|[ 0-9_])", '', vendor_raw.lower())
                    # vendor = str(userid) + '-' + vendor_txt
                    vendor = entity + '-' + vendor_txt
                else:
                    vendor = ''
                exp_name = p[10]
                exp_key = p[11]
                ds_req = p[12]

                if len(ds_req) < 2:
                    no_ds_request += 1
                    continue

                v_list = [0 for _ in range(4*self.num_slots+self.extra_slots)]
                self.extra_slot_labels = []
                if entity in comp_hist.keys():
                    this_comp_hist = comp_hist[entity].most_common(self.num_slots)
                    list_labels = [x[0] for x in this_comp_hist]
                    list_labels.extend([0 for _ in range(self.num_slots - len(list_labels))])
                    v_list = self.make_vector(v_list, 0, list_labels, comp_hist[entity])
                else:
                    no_entity_history += 1
                    continue
                this_user_hist = user_hist[userid] if userid in user_hist.keys() else Counter()
                v_list = self.make_vector(v_list, 1, list_labels, this_user_hist)
                this_vend_hist = vend_hist[vendor] if vendor in vend_hist.keys() else Counter()
                v_list = self.make_vector(v_list, 2, list_labels, this_vend_hist)
                if amount and entity in amt_hist.keys():
                    amount = float(amount)
                    if amount == 0:
                        this_amnt_hist = amt_hist[entity]['Zero']
                    elif amount < 10:
                        this_amnt_hist = amt_hist[entity]['Under Ten']
                    elif amount < 50:
                        this_amnt_hist = amt_hist[entity]['Under Fifty']
                    elif amount < 100:
                        this_amnt_hist = amt_hist[entity]['Under Hundred']
                    elif amount < 1000:
                        this_amnt_hist = amt_hist[entity]['Under Thousand']
                    else:
                        this_amnt_hist = amt_hist[entity]['Other']
                else:
                    this_amnt_hist = Counter()
                v_list = self.make_vector(v_list, 3, list_labels, this_amnt_hist)

                if exp_key in list_labels:
                    self.y_train.append(list_labels.index(exp_key))
                elif exp_key in self.extra_slot_labels:
                    in_extra_slots += 1
                    self.y_train.append(len(list_labels)+self.extra_slot_labels.index(exp_key))
                else:
                    too_rare += 1
                    continue
                self.x_train.append(v_list)
                succeeded += 1
        pickle.dump(self.x_train, open('stackedmodel/x_train.pkl', 'wb'))
        pickle.dump(self.y_train, open('stackedmodel/y_train.pkl', 'wb'))

        print("no ds request: ", no_ds_request)
        print("no entity history: ", no_entity_history)
        print("in extra slots: ", in_extra_slots)
        print("too rare (not in any slots): ", too_rare)
        print("succeeded: ", succeeded)

    def run_training(self):
        self.x_train = pickle.load(open('stackedmodel/x_train.pkl', 'rb'))
        self.y_train = pickle.load(open('stackedmodel/y_train.pkl', 'rb'))

        clf = linear_model.LogisticRegression(random_state=1, solver='liblinear')
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open('stackedmodel/clf.pkl', 'wb'))

if __name__ == '__main__':
    start = time.time()
    new_train = StackTrainer()
    new_train.create_train_files()
    new_train.run_training()
    end = time.time()

    print("time: ", str(end-start))
