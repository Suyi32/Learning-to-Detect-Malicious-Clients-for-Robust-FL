#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

gpu = 0
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and gpu != -1 else 'cpu')

def test_img_poison(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    if args.dataset == "mnist":
        correct  = torch.tensor([0.0] * 10)
        gold_all = torch.tensor([0.0] * 10)
    elif args.dataset == "femnist":
        correct  = torch.tensor([0.0] * 62)
        gold_all = torch.tensor([0.0] * 62)
    else:
        print("Unknown dataset")
        exit(0)

    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        y_gold = target.data.view_as(y_pred).squeeze(1)
        y_pred = y_pred.squeeze(1)

        for pred_idx in range(len(y_pred)):
            gold_all[ y_gold[pred_idx] ] += 1
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            elif y_pred[pred_idx] == 5 and y_gold[pred_idx] == 7:  # poison attack
                poison_correct += 1

    test_loss /= len(data_loader.dataset)

    accuracy = 100.00 * (sum(correct) / sum(gold_all)).item()
    acc_per_label = correct / gold_all

    return accuracy, test_loss, acc_per_label.tolist(), poison_correct/gold_all[7].item()

    # if args.verbose:
    #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #         test_loss, correct, len(data_loader.dataset), accuracy))
    # return accuracy, test_loss

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss