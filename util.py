import xlwt



def init_log(client_n):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('results',cell_overwrite_ok=True)
    row = ['settings']
    # row += '-'
    # row.append(['local_train_acc'])
    # row.append(['local_train_loss'])

    for i in range(client_n):
        row += '-'
        row.append(['test_acc_{}'.format(i)])

    row += '-'
    row += '-'
    for i in range(client_n):
        row.append(['local_train_loss_{}'.format(i)])
    

    style = xlwt.XFStyle()
    style.alignment.wrap = 1

    for i in range(len(row)):
        sheet1.write(0, i, row[i])

    return f, sheet1