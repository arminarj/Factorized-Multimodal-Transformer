import json
import os
import signal
import sys
import time
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score

from consts import global_consts as gc
from model import Net



def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def correlation(X, Y):
    return np.mean(((X - np.mean(X)) * (Y - np.mean(Y)))) / (np.var(X) ** 0.5) / (np.var(Y) ** 0.5)


def eval_mosi(split, output_all, label_all):
    truth = np.array(label_all)
    preds = np.array(output_all)
    mae = np.mean(np.abs(truth - preds))
    acc = accuracy_score(truth >= 0, preds >= 0)
    corr = np.corrcoef(preds, truth)[0][1]
    non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])

    preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    truth_a7 = np.clip(truth, a_min=-3., a_max=3.)
    preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    truth_a5 = np.clip(truth, a_min=-2., a_max=2.)
    acc_7 = multiclass_acc(preds_a7, truth_a7)
    acc_5 = multiclass_acc(preds_a5, truth_a5)
    f1_mfn = f1_score(np.round(truth), np.round(preds), average="weighted")
    f1_raven = f1_score(truth >= 0, preds >= 0, average="weighted")
    f1_muit = f1_score((preds[non_zeros] > 0), (truth[non_zeros] > 0), average='weighted')
    binary_truth = (truth[non_zeros] > 0)
    binary_preds = (preds[non_zeros] > 0)
    ex_zero_acc = accuracy_score(binary_truth, binary_preds)
    print("\t%s mean error: %f" % (split, mae))
    print("\t%s correlation coefficient: %f" % (split, corr))
    print("\t%s accuracy: %f" % (split, acc))
    print("\t%s mult_acc_7: %f" % (split, acc_7))
    print("\t%s mult_acc_5: %f" % (split, acc_5))
    print("\t%s F1 MFN: %f " % (split, f1_mfn))
    print("\t%s F1 RAVEN: %f " % (split, f1_raven))
    print("\t%s F1 MuIT: %f " % (split, f1_muit))
    print("\t%s exclude zero accuracy: %f" % (split, ex_zero_acc))
    return mae, corr, acc, acc_7, acc_5, f1_mfn, f1_raven, f1_muit, ex_zero_acc

    def eval_mosei_emo(split, output_all, label_all):
        truth = np.array(label_all)
        preds = np.array(output_all)
        mae = np.mean(np.abs(truth - preds))
        mse = np.sqrt(np.sum(np.square(truth - preds)))
        # acc = accuracy_score(truth >= 0, preds >= 0)
        # corr = np.corrcoef(preds, truth)[0][1]
        # non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])

        # preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
        # truth_a7 = np.clip(truth, a_min=-3., a_max=3.)
        # preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
        # truth_a5 = np.clip(truth, a_min=-2., a_max=2.)
        # acc_7 = multiclass_acc(preds_a7, truth_a7)
        # acc_5 = multiclass_acc(preds_a5, truth_a5)
        # f1_mfn = f1_score(np.round(truth), np.round(preds), average="weighted")
        # f1_raven = f1_score(truth >= 0, preds >= 0, average="weighted")
        # f1_muit = f1_score((preds[non_zeros] > 0), (truth[non_zeros] > 0), average='weighted')
        # binary_truth = (truth[non_zeros] > 0)
        # binary_preds = (preds[non_zeros] > 0)
        # ex_zero_acc = accuracy_score(binary_truth, binary_preds)
        print("\t%s mean error: %f" % (split, mae))
        print("\t%s mean square error: %f" % (split, mse))
        print("\t%s correlation coefficient: %f" % (split, corr))
        # print("\t%s accuracy: %f" % (split, acc))
        # print("\t%s mult_acc_7: %f" % (split, acc_7))
        # print("\t%s mult_acc_5: %f" % (split, acc_5))
        # print("\t%s F1 MFN: %f " % (split, f1_mfn))
        # print("\t%s F1 RAVEN: %f " % (split, f1_raven))
        # print("\t%s F1 MuIT: %f " % (split, f1_muit))
        # print("\t%s exclude zero accuracy: %f" % (split, ex_zero_acc))
        return mae, corr, mse


def eval_iemocap(split, output_all, label_all):
    truths = np.array(label_all)
    results = np.array(output_all)
    test_preds = results.reshape((-1, 4, 2))
    test_truth = truths.reshape((-1, 4))
    emos_f1 = {}
    emos_acc = {}
    for emo_ind, em in enumerate(gc.best.emos):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        emos_f1[em] = f1
        acc = accuracy_score(test_truth_i, test_preds_i)
        emos_acc[em] = acc
        print("\t%s %s F1 Score: %f" % (split, gc.best.emos[emo_ind], f1))
        print("\t%s %s Accuracy: %f" % (split, gc.best.emos[emo_ind], acc))
    return emos_f1, emos_acc


def eval_pom(split, output_all, label_all):
    truths = np.array(label_all).reshape((-1, len(gc.best.pom_cls)))
    preds = np.array(output_all).reshape((-1, len(gc.best.pom_cls)))
    cls_mae, cls_corr, cls_acc = {}, {}, {}
    for cls_id, cls in enumerate(gc.best.pom_cls):
        mae = np.mean(np.absolute(preds[:, cls_id] - truths[:, cls_id]))
        corr = np.corrcoef(preds[:, cls_id], truths[:, cls_id])[0][1]
        acc = np.mean(np.round(preds[:, cls_id]) == np.round(truths[:, cls_id]))
        cls_mae[cls], cls_corr[cls], cls_acc[cls] = mae, corr, acc
        print("\t%s %s mae: %f" % (split, cls, round(mae, 5)))
        print("\t%s %s corr: %f" % (split, cls, round(corr, 5)))
        print("\t%s %s mult_acc: %f" % (split, cls, round(acc, 5)))
    return cls_mae, {'corr': cls_corr, 'acc': cls_acc}


def logSummary():
    print("best epoch: %d" % gc.best.best_epoch)
    if gc.dataset == "iemocap":
        for split in ["test", "valid", "test_at_valid_max"]:
            for em in gc.best.emos:
                print("highest %s %s F1: %f" % (split, em, gc.best.max_f1[split][em]))
                print("highest %s %s accuracy: %f" % (split, em, gc.best.max_acc[split][em]))
    elif gc.dataset == 'pom':
        for split in gc.best.split:
            for cls in gc.best.pom_cls:
                for metric in ['corr', 'acc']:
                    print("highest %s %s %s: %f" % (split, cls, metric, gc.best.max_pom_metrics[metric][split][cls]))
                print("best %s MAE %s: %f" % (split, cls, gc.best.best_pom_mae[split][cls]))

    # elif gc.dataset == 'mosi':
    #     print("best epoch: %d" % gc.best.best_epoch)
    #     print("lowest training MAE: %f" % gc.best.min_train_mae)
    #     print("lowest testing MAE: %f" % gc.best.min_test_mae)
    #     print("lowest validation MAE: %f" % gc.best.min_valid_mae)
    #     print("test MAE when validation MAE is the lowest: %f" % gc.best.test_mae_at_valid_min)

    #     print("highest testing F1 MFN: %f" % gc.best.max_test_f1_mfn)
    #     print("highest testing F1 RAVEN: %f" % gc.best.max_test_f1_raven)
    #     print("highest testing F1 MuIT: %f" % gc.best.max_test_f1_muit)

    #     print("highest validation F1 MFN: %f" % gc.best.max_valid_f1_mfn)
    #     print("highest validation F1 RAVEN: %f" % gc.best.max_valid_f1_raven)
    #     print("highest validation F1 MuIT: %f" % gc.best.max_valid_f1_muit)

    #     print("test F1 MFN when validation F1 is the highest: %f" % gc.best.test_f1_mfn_at_valid_max)
    #     print("test F1 RAVEN when validation F1 is the highest: %f" % gc.best.test_f1_raven_at_valid_max)
    #     print("test F1 MuIT when validation F1 is the highest: %f" % gc.best.test_f1_muit_at_valid_max)

    #     print("highest testing correlation: %f" % gc.best.max_test_cor)
    #     print("highest validation correlation: %f" % gc.best.max_valid_cor)
    #     print("test correlation when validation correlation is the highest: %f" % gc.best.test_cor_at_valid_max)

    #     print("highest testing accuracy: %f" % gc.best.max_test_acc)
    #     print("highest validation accuracy: %f" % gc.best.max_valid_acc)
    #     print("test accuracy when validation accuracy is the highest: %f" % gc.best.test_acc_at_valid_max)

    #     print("highest testing exclude zero accuracy: %f" % gc.best.max_test_ex_zero_acc)
    #     print("highest validation exclude zero accuracy: %f" % gc.best.max_valid_ex_zero_acc)
    #     print("test ex-zero accuracy when validation ex-zero accuracy is the highest: %f" %
    #           gc.best.test_ex_zero_acc_at_valid_max)

    #     print("highest testing accuracy 5: %f" % gc.best.max_test_acc_5)
    #     print("highest validation accuracy 5: %f" % gc.best.max_valid_acc_5)
    #     print("test accuracy 5 when validation accuracy 5 is the highest: %f" % gc.best.test_acc_5_at_valid_max)

    #     print("highest testing accuracy 7: %f" % gc.best.max_test_acc_7)
    #     print("highest validation accuracy 7: %f" % gc.best.max_valid_acc_7)
    #     print("test accuracy 7 when validation accuracy 7 is the highest: %f" % gc.best.test_acc_7_at_valid_max)
    else :
        print(f"mae : valid {gc.best.max_mae['valid']}, test, {gc.best.max_mae['test_at_valid_max']} ") 
        print(f"corr : valid {gc.best.max_corr['valid']} , test, {gc.best.max_corr['test_at_valid_max']} ") 
        print(f"corr : valid {gc.best.max_mse['valid']} , test, {gc.best.max_mse['test_at_valid_max']} ") 


def stopTraining(signum, frame):
    global savedStdout
    logSummary()
    sys.stdout = savedStdout
    sys.exit()


def train_model(config_file_name, model_name):
    save_epochs = [1, 10, 50, 100, 150, 200, 500, 700, 999]
    config_name = ''
    if config_file_name:
        config_name = os.path.splitext(os.path.basename(config_file_name))[0]
    if model_name is None:
        model_name = config_name
    try:
        signal.signal(signal.SIGINT, stopTraining)
        signal.signal(signal.SIGTERM, stopTraining)
    except:
        pass

    global savedStdout
    savedStdout = sys.stdout

    if gc.log_path != None:
        dir_path = "%s%d" % (gc.log_path, gc.HPID)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        log_file = "%s/print.log" % dir_path
        f = open(log_file, "w+")
        sys.stdout = f
    if gc.dataset == "mosi_short":
        from MOSI_dataset import MOSIDataset
        ds = MOSIDataset
    elif gc.dataset == 'mosei_new':
        from MOSEI_new_dataset import MoseiNewDataset
        ds = MoseiNewDataset
    elif gc.dataset == 'mosi1' or gc.dataset == 'mosi2':
        from MOSI_new_dataset import MosiNewDataset
        ds = MosiNewDataset
    elif gc.dataset == 'pom':
        from POM_dataset import PomDataset
        ds = PomDataset
    else:
        from Multimodal_dataset import MultimodalDataset
        ds = MultimodalDataset

    train_dataset = ds(gc.data_path, cls="train")
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=gc.batch_size,
        shuffle=True,
        num_workers=1,
    )

    test_dataset = ds(gc.data_path, cls="test")
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=gc.batch_size,
        shuffle=False,
        num_workers=1,
    )

    valid_dataset = ds(gc.data_path, cls="valid")
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=gc.batch_size,
        shuffle=False,
        num_workers=1,
    )

    print("HPID:%d:Data Successfully Loaded." % gc.HPID)

    if gc.single_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        device = torch.device("cuda:%d" % gc.config['cuda'] if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gc.config['cuda'])
    gc.device = device
    print("running device: ", device)
    gc().logParameters()

    net = Net()
    print(net)
    net.to(device)

    if gc.dataset == "iemocap":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': net.proj_a.parameters()},
        {'params': net.proj_v.parameters()},
        {'params': net.transformer_encoder.parameters()},
        {'params': net.finalW.parameters()},
        {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
    ], betas=(0.9, 0.98), eps=1e-09, lr=gc.config['lr'])
    start_epoch = 0
    model_path = os.path.join(gc.model_path, gc.dataset + '_' + model_name + '.tar')
    if gc.load_model and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        gc.best = checkpoint['best']

    running_loss = 0.0
    for epoch in range(start_epoch, gc.config['epoch_num']):
        if epoch % 10 == 0:
            print("HPID:%d:Training Epoch %d." % (gc.HPID, epoch))
        if gc.save_grad and epoch in save_epochs:
            grad_dict = {}
            update_dict = {}
        if epoch % 100 == 0:
            logSummary()
        if gc.lr_decay and (epoch == 75 or epoch == 200):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        with torch.no_grad():
            print("Epoch #%d results:" % epoch)
            test_label_all = []
            test_output_all = []
            for data in test_loader:
                words, covarep, facet, inputLen, labels = data
                if covarep.size()[0] == 1:
                    continue
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = net(words, covarep, facet, inputLen)
                if gc.dataset == 'iemocap':
                    outputs = outputs.view(-1, 2)
                    labels = labels.view(-1)
                test_output_all.extend(outputs.tolist())
                test_label_all.extend(labels.tolist())
            if gc.dataset == "iemocap":
                test_f1, test_acc = eval_iemocap('test', test_output_all, test_label_all)
            elif gc.dataset == "pom":
                test_mae, test_metrics = eval_pom('test', test_output_all, test_label_all)
            elif gc.dataset == "mosi":
                test_mae, test_cor, test_acc, test_acc_7, test_acc_5, test_f1_mfn, test_f1_raven, test_f1_muit, \
                test_ex_zero_acc = eval_mosi('test', test_output_all, test_label_all)
            else:
                test_mae, test_corr, test_mse = eval_mosei_emo('test', test_output_all, test_label_all)

            label_all = []
            output_all = []
            for data in valid_loader:
                words, covarep, facet, inputLen, labels = data
                if covarep.size()[0] == 1:
                    continue
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = net(words, covarep, facet, inputLen)
                if gc.dataset == 'iemocap':
                    outputs = outputs.view(-1, 2)
                    labels = labels.view(-1)
                output_all.extend(outputs.data.cpu().tolist())
                label_all.extend(labels.data.cpu().tolist())
            if gc.dataset == "iemocap":
                valid_f1, valid_acc = eval_iemocap('valid', output_all, label_all)
                for em in gc.best.emos:
                    if valid_f1[em] > gc.best.max_f1['valid'][em]:
                        gc.best.max_f1['valid'][em] = valid_f1[em]
                        gc.best.max_f1['test_at_valid_max'][em] = test_f1[em]
                    if valid_acc[em] > gc.best.max_acc['valid'][em]:
                        gc.best.max_acc['valid'][em] = valid_acc[em]
                        gc.best.max_acc['test_at_valid_max'][em] = test_acc[em]
                    if test_f1[em] > gc.best.max_f1['test'][em]:
                        gc.best.max_f1['test'][em] = test_f1[em]
                        gc.best.best_epoch = epoch + 1
                    if test_acc[em] > gc.best.max_acc['test'][em]:
                        gc.best.max_acc['test'][em] = test_acc[em]
            elif gc.dataset == "pom":
                valid_mae, valid_metrics = eval_pom('valid', output_all, label_all)
                for cls in gc.best.pom_cls:
                    if valid_mae[cls] < gc.best.best_pom_mae['valid'][cls]:
                        gc.best.best_pom_mae['valid'][cls] = valid_mae[cls]
                        gc.best.best_pom_mae['test_at_valid_max'][cls] = test_mae[cls]
                    if test_mae[cls] < gc.best.best_pom_mae['test'][cls]:
                        gc.best.best_pom_mae['test'][cls] = test_mae[cls]
                        gc.best.best_epoch = epoch + 1
                        best_model = True
                    for metric in ['acc', 'corr']:
                        if valid_metrics[metric][cls] > gc.best.max_pom_metrics[metric]['valid'][cls]:
                            gc.best.max_pom_metrics[metric]['valid'][cls] = valid_metrics[metric][cls]
                            gc.best.max_pom_metrics[metric]['test_at_valid_max'][cls] = test_metrics[metric][cls]
                        if test_metrics[metric][cls] > gc.best.max_pom_metrics[metric]['test'][cls]:
                            gc.best.max_pom_metrics[metric]['test'][cls] = test_metrics[metric][cls]
            # elif gc.dataset == "mosi":
            #     if len(output_all) > 0:
            #         valid_mae, valid_cor, valid_acc, valid_acc_7, valid_acc_5, valid_f1_mfn, valid_f1_raven, \
            #         valid_f1_muit, valid_ex_zero_acc = eval_mosi('valid', output_all, label_all)
            #         if valid_mae < gc.best.min_valid_mae:
            #             gc.best.min_valid_mae = valid_mae
            #             gc.best.test_mae_at_valid_min = test_mae
            #         if valid_cor > gc.best.max_valid_cor:
            #             gc.best.max_valid_cor = valid_cor
            #             gc.best.test_cor_at_valid_max = test_cor
            #         if valid_acc > gc.best.max_valid_acc:
            #             gc.best.max_valid_acc = valid_acc
            #             gc.best.test_acc_at_valid_max = test_acc
            #         if valid_ex_zero_acc > gc.best.max_valid_ex_zero_acc:
            #             gc.best.max_valid_ex_zero_acc = valid_ex_zero_acc
            #             gc.best.test_ex_zero_acc_at_valid_max = test_ex_zero_acc
            #         if valid_acc_5 > gc.best.max_valid_acc_5:
            #             gc.best.max_valid_acc_5 = valid_acc_5
            #             gc.best.test_acc_5_at_valid_max = test_acc_5
            #         if valid_acc_7 > gc.best.max_valid_acc_7:
            #             gc.best.max_valid_acc_7 = valid_acc_7
            #             gc.best.test_acc_7_at_valid_max = test_acc_7

            #         if valid_f1_mfn > gc.best.max_valid_f1_mfn:
            #             gc.best.max_valid_f1_mfn = valid_f1_mfn
            #             gc.best.test_f1_mfn_at_valid_max = test_f1_mfn
            #         if valid_f1_raven > gc.best.max_valid_f1_raven:
            #             gc.best.max_valid_f1_raven = valid_f1_raven
            #             gc.best.test_f1_raven_at_valid_max = test_f1_raven
            #         if valid_f1_muit > gc.best.max_valid_f1_muit:
            #             gc.best.max_valid_f1_muit = valid_f1_muit
            #             gc.best.test_f1_muit_at_valid_max = test_f1_muit

            #         if test_mae < gc.best.min_test_mae:
            #             gc.best.min_test_mae = test_mae
            #             gc.best.best_epoch = epoch + 1
            #             best_model = True
            #         if test_cor > gc.best.max_test_cor:
            #             gc.best.max_test_cor = test_cor
            #         if test_acc > gc.best.max_test_acc:
            #             gc.best.max_test_acc = test_acc
            #         if test_ex_zero_acc > gc.best.max_test_ex_zero_acc:
            #             gc.best.max_test_ex_zero_acc = test_ex_zero_acc
            #         if test_acc_5 > gc.best.max_test_acc_5:
            #             gc.best.max_test_acc_5 = test_acc_5
            #         if test_acc_7 > gc.best.max_test_acc_7:
            #             gc.best.max_test_acc_7 = test_acc_7
            #         if test_f1_mfn > gc.best.max_test_f1_mfn:
            #             gc.best.max_test_f1_mfn = test_f1_mfn
            #         if test_f1_raven > gc.best.max_test_f1_raven:
            #             gc.best.max_test_f1_raven = test_f1_raven
            #         if test_f1_muit > gc.best.max_test_f1_muit:
            #             gc.best.max_test_f1_muit = test_f1_muit
            else :
                valid_mae, valid_corr, valid_mse = eval_mosei_emo('valid', output_all, label_all)
                if valid_mae > gc.best.max_mae['valid']:
                    gc.best.max_mae['valid'] = valid_mae
                    gc.best.max_mae['test_at_valid_max'] = test_mae
                if valid_corr > gc.best.max_acc['valid']:
                    gc.best.max_corr['valid'] = valid_corr
                    gc.best.max_corr['test_at_valid_max'] = test_corr
                if valid_mse > gc.best.max_mse['valid']:
                    gc.best.max_mse['valid'] = valid_mse
                    gc.best.max_mse['test_at_valid_max'] = test_mse

        tot_num = 0
        tot_err = 0
        tot_right = 0
        label_all = []
        output_all = []
        max_i = 0
        for i, data in enumerate(train_loader):
            batch_update_dict = {}
            max_i = i
            words, covarep, facet, inputLen, labels = data
            if covarep.size()[0] == 1:
                continue
            words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                device), inputLen.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(words, covarep, facet, inputLen)
            output_all.extend(outputs.tolist())
            label_all.extend(labels.tolist())
            if gc.dataset != "iemocap" and gc.dataset != "pom":
                err = torch.sum(torch.abs(outputs - labels))
                tot_right += torch.sum(torch.eq(torch.sign(labels), torch.sign(outputs)))
                tot_err += err
                tot_num += covarep.size()[0]
            if gc.dataset == 'iemocap':
                outputs = outputs.view(-1, 2)
                labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gc.config['max_grad'], norm_type=inf)
            if gc.save_grad and epoch in save_epochs:
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    try:
                        if i == 0:
                            grad_dict[name] = param.grad.detach().cpu().numpy()
                        else:
                            grad_dict[name] = grad_dict[name] + np.abs(param.grad.detach().cpu().numpy())
                        assert (name not in batch_update_dict)
                        batch_update_dict[name] = param.data.detach().cpu().numpy()
                    except:
                        import pdb
                        pdb.set_trace()
            optimizer.step()
            if gc.save_grad and epoch in save_epochs:
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    if i == 0:
                        update_dict[name] = np.abs(batch_update_dict[name] - param.data.detach().cpu().numpy())
                    else:
                        update_dict[name] += np.abs(batch_update_dict[name] - param.data.detach().cpu().numpy())
            running_loss += loss.item()
            del loss
            del outputs
            if i % 20 == 19:
                torch.cuda.empty_cache()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        if gc.dataset == "iemocap":
            eval_iemocap('train', output_all, label_all)
        elif gc.dataset == 'pom':
            eval_pom('train', output_all, label_all)
        else:
            train_mae = tot_err / tot_num
            train_acc = float(tot_right) / tot_num
            print("\ttrain mean error: %f" % train_mae)
            print("\ttrain acc: %f" % train_acc)
            if train_mae < gc.best.min_train_mae:
                gc.best.min_train_mae = train_mae
            if train_acc > gc.best.max_train_acc:
                gc.best.max_train_acc = train_acc

        if gc.save_grad and epoch in save_epochs:
            grad_f = h5py.File(os.path.join(gc.model_path, '%s_grad_%s_%d.hdf5' % (gc.dataset, config_name, epoch)))
            update_f = h5py.File(os.path.join(gc.model_path, '%s_update_%s_%d.hdf5' % (gc.dataset, config_name, epoch)))
            for name in grad_dict.keys():
                grad_avg = grad_dict[name] / (max_i + 1)
                grad_f.create_dataset(name, data=grad_avg)
                update_avg = update_dict[name] / (max_i + 1)
                update_f.create_dataset(name, data=update_avg)
            grad_f.close()
            update_f.close()

    logSummary()


if __name__ == "__main__":
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
    config_file_name = ''
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        gc.config = json.load(open(config_file_name), object_pairs_hook=OrderedDict)
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    else:
        model_name = None
    torch.manual_seed(gc.config['seed'])
    train_model(config_file_name, model_name)
    elapsed_time = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
