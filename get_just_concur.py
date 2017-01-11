file_name = 'all_trainset'
new_file = open('voith_trainset', 'w')
with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        p = line.split('\t')
        entity = p[0]
        userid = p[1]
        datekey = int(p[2])
        expense_key = p[3]
        expense_name = p[4]
        longer_type = True if len(p) > 6 else False
        if entity != 'p0000745jr8u':  # 'p00425z4gu':
            continue
        amount = p[5]
        vendor = p[6]
        expenseit = p[7]
        ocr = p[8]
        new_file.write(line)
new_file.close()