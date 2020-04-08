from consts import global_consts as gc
import os
import json
from collections import OrderedDict
import csv


def get_metrics(fname):
    lines = 200
    fsize = os.stat(fname).st_size
    bufsize = 4048
    iter = 0
    best_epoch = None
    if gc.dataset == "iemocap":
        max_f1 = {"Neutral": 0, "Happy": 0, "Sad": 0, "Angry": 0}
        max_acc = {"Neutral": 0, "Happy": 0, "Sad": 0, "Angry": 0}
    elif gc.dataset == "pom":
        max_pom_metrics = {metric: {} for metric in ['acc', 'corr']}
        for metric in ['acc', 'corr']:
            for cls in gc.best.pom_cls:
                max_pom_metrics[metric][cls] = 0

        best_pom_mae = {}
        for cls in gc.best.pom_cls:
            best_pom_mae[cls] = 10
    else:
        mae, corr, best_epoch, f1_mfn, f1_raven, f1_muit, acc, acc_ex_zero = None, None, None, None, None, None, None, None

    with open(fname) as f:
        if bufsize > fsize:
            bufsize = fsize - 1
        data = []
        while True:
            iter += 1
            f.seek(max(fsize - bufsize * iter, 0))
            data.extend(f.readlines())
            if len(data) >= lines or f.tell() == 0:
                data = data[-lines:]
                break
        for line in data:
            line_start = 'best epoch: '
            if line.startswith(line_start):
                best_epoch = int(line.split(line_start)[1])
                continue
            if gc.dataset == "iemocap":
                for em in gc.best.iemocap_emos:
                    line_start = "highest test %s F1: " % (em)
                    if line.startswith(line_start):
                        max_f1[em] = float(line.split(line_start)[1])
                        break
                    line_start = "highest test %s accuracy: " % (em)
                    if line.startswith(line_start):
                        max_acc[em] = float(line.split(line_start)[1])
                        break
            elif gc.dataset == "pom":
                for cls in gc.best.pom_cls:
                    found = False
                    for metric in ['corr', 'acc']:
                        line_start = "highest test %s %s: " % (cls, metric)
                        if line.startswith(line_start):
                            max_pom_metrics[metric][cls] = float(line.split(line_start)[1])
                            found = True
                            break
                    if found:
                        break
                    line_start = "best test MAE %s: " % (cls)
                    if line.startswith(line_start):
                        best_pom_mae[cls] = float(line.split(line_start)[1])
                    break
            else:
                line_start = 'lowest testing MAE: '
                if line.startswith(line_start):
                    mae = float(line.split(line_start)[1])
                    continue
                line_start = 'highest testing correlation: '
                if line.startswith(line_start):
                    corr = float(line.split(line_start)[1])
                    continue
                line_start = 'highest testing F1 MFN: '
                if line.startswith(line_start):
                    f1_mfn = float(line.strip().split(line_start)[1])
                    continue
                line_start = 'highest testing F1 RAVEN: '
                if line.startswith(line_start):
                    f1_raven = float(line.strip().split(line_start)[1])
                    continue
                line_start = 'highest testing F1 MuIT: '
                if line.startswith(line_start):
                    f1_muit = float(line.strip().split(line_start)[1])
                    continue
                line_start = 'highest testing accuracy: '
                if line.startswith(line_start):
                    acc = float(line.strip().split(line_start)[1])
                    continue
                line_start = 'highest testing exclude zero accuracy: '
                if line.startswith(line_start):
                    acc_ex_zero = float(line.strip().split(line_start)[1])
                    continue

    if gc.dataset == "iemocap":
        return best_epoch, max_f1, max_acc
    elif gc.dataset == "pom":
        return best_epoch, best_pom_mae, max_pom_metrics
    else:
        mae, corr, best_epoch, f1_mfn, f1_raven, f1_muit, acc, acc_ex_zero


LOG_DIR = '../logs/'
CONF_DIR = '../configs/'
count = 0
lines = []
headers = []
pom_cls = gc.best.pom_cls.copy()
pom_header = ["Con", "Pas", "Voi", "Dom", "Cre", "Viv", "Exp",
              "Ent", "Res", "Tru", "Rel", "Out", "Tho",
              "Ner", "Per", "Hum"]
pom_cls.remove('Sentiment')
for log_file in os.listdir(LOG_DIR):
    config_name = os.path.splitext(os.path.basename(log_file))[0]
    config_file = CONF_DIR + config_name + '.json'
    if not os.path.exists(config_file):
        continue
    if gc.dataset == "iemocap":
        best_epoch, max_f1, max_acc = get_metrics(LOG_DIR + log_file)
    elif gc.dataset == "pom":
        best_epoch, best_pom_mae, max_pom_metrics = get_metrics(LOG_DIR + log_file)
    else:
        mae, corr, best_epoch, f1_mfn, f1_raven, f1_muit, acc, acc_ex_zero = get_metrics(LOG_DIR + log_file)
    if best_epoch is None:
        print('Cannot find mae or corr in %s' % LOG_DIR + log_file)
        continue

    config = json.load(open(config_file), object_pairs_hook=OrderedDict)
    if count == 0:
        headers = list(config.keys())
        if gc.dataset == "iemocap":
            lines.append(['config'] + headers + ['best epoch'] + ['%s %s' % (em, metric) for em in gc.best.iemocap_emos
                                                                  for metric in ['acc', 'F1']])
        elif gc.dataset == "pom":
            lines.append(['config'] + headers + ['best epoch'] +
                         ['%s acc' % cl for cl in pom_header] +
                         ['%s mae' % cl for cl in pom_header] +
                         ['%s corr' % cl for cl in pom_header])
        else:
            lines.append(['config'] + headers + ['MAE', 'corr', 'best epoch', 'F1 MFN', 'F1 RAVEN', 'F1 MuIT',
                                                 'Acc', 'Acc ex-0'])
        count += 1
    if gc.dataset == "iemocap":
        lines.append([config_file] + list(config.values()) + [best_epoch] + [round(metric[em] * 100, 1)
                                                                             for em in gc.best.iemocap_emos
                                                                             for metric in [max_acc, max_f1]])
    elif gc.dataset == "pom":
        lines.append([config_file] + list(config.values()) + [best_epoch] +
                     [round(max_pom_metrics['acc'][cl] * 100, 1) for cl in pom_cls] +
                     [round(best_pom_mae[cl], 3) for cl in pom_cls] +
                     [round(max_pom_metrics['corr'][cl], 3) for cl in pom_cls])
    else:
        lines.append([config_file] + list(config.values()) +
                     [mae, corr, best_epoch, f1_mfn, f1_raven, f1_muit, acc, acc_ex_zero])

with open('./result.csv', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(lines)