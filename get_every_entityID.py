import pickle
import time

start = time.time()
file_name = 'all_trainset'

company_list = []

with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        p = line.split('\t')
        entity = p[0]
        if entity not in company_list:
            company_list.append(entity)

pickle.dump(company_list, open('inputdata/all_company_list.pkl', 'wb'))
end = time.time()
print("success. time = ", str(end-start))

