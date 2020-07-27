def print_log_headers(fp, train_logs, valid_logs):
    fp.write('epoch,')
    for k in train_logs.keys():
        fp.write('train_' + k + ',')
    for k in valid_logs.keys():
        fp.write('val_' + k + ',')
    fp.write('f1score,tmc,pmc\n')

    fp.flush()


def print_log_lines(fp, itera, train_logs, valid_logs, f1score, tmc, pmc):
    fp.write('{:d},'.format(itera))
    for k in train_logs.keys():
        fp.write('{:4f},'.format(train_logs[k]))
    for k in valid_logs.keys():
        fp.write('{:4f},'.format(valid_logs[k]))
    fp.write('{:.4f},{:.2f},{:.2f}\n'.format(f1score, tmc, pmc))

    fp.flush()


from io import StringIO


def print_log_lines_console(fp, itera, train_logs, valid_logs, f1score, tmc, pmc):
    output = StringIO()
    output.write('{:d},'.format(itera))
    for k in train_logs.keys():
        output.write('{:4f},'.format(train_logs[k]))
    for k in valid_logs.keys():
        output.write('{:4f},'.format(valid_logs[k]))
    output.write('{:.4f},{:.2f},{:.2f}\n'.format(f1score, tmc, pmc))

    ret = output.getvalue()
    print(ret)
    output.close()

