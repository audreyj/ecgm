import pickle
import time
import urllib.request
import json

company_list = ['p0000745jr8u', 'p0003884x7lt', 'p0043611aoji', 'p0039557fvbf',
                'p0090051h2oq', 'p0006679vz2s', 'p00425z4gu', 'p0009976ed7k',
                'p0005859huep', 'p0089280mxzg', 'p0079383unbs', 'p00521862bvn',
                'p0002406hlem', 'p0002233vdsm', 'p0096373arhm', 'p0039096va39',
                'p0046598vm29', 'p0036394dcvf', 'p0081759udz2', 'p0079434mdig',
                'p0014655asb9', 'p0043883wrzh']


def prod_check(entity):

    x_test = pickle.load(open("intermediate/" + entity + "_x_test.pkl", "rb"))
    y_test = pickle.load(open("intermediate/" + entity + "_y_testET.pkl", "rb"))
    test_info = pickle.load(open('intermediate/' + entity + '_info_test.pkl', 'rb'))

    # Call API running in prod
    ds_api_url = 'http://seapr1dsweb.concurasp.com:80/ds-webapi/service/expenseClassification/receiptTypeClassification'
    # Call API running in RQA
    # ds_api_url = 'http://10.24.25.120:80/ds-webapi/service/expenseClassification/receiptTypeClassification'
    request_type = {'Content-Type': 'application/json'}

    correct_count = 0
    call_count = 0
    # Call the API with each ds_request and store the result in the ds_response column
    for i, ocr in enumerate(x_test):
        data = {"entityId": entity, "ocrText": ocr, "userId": test_info[i]['userid']}
        data = json.dumps(data).encode('utf-8', 'ignore')

        call_count += 1
        # if call_count > 100:
        #     break

        req = urllib.request.Request(ds_api_url, data, request_type)
        f = urllib.request.urlopen(req)

        ds_response = json.loads(f.read().decode('utf-8'))
        pred = ds_response['expenseTypes'][0]['type']
        print(y_test[i], " || ", pred)
        if pred == y_test[i]:
            correct_count += 1

    print("%s Accuracy: %0.3f" % (entity, (float(correct_count)) / call_count))

if __name__ == "__main__":
    for company in company_list:
        prod_check(company)
        temp = input("pause")
