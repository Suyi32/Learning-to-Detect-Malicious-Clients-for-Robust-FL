#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn

from utils.sampling import mnist_noniid
import argparse
from datetime import datetime
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img, test_img_poison
from models.Nets import LogisticRegression

from attacks import sign_flipping_attack, additive_noise
from aggregations import aggregation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=350, help="rounds of training")

    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--sample_users', type=int, default=100, help="number of users in federated learning C")
    parser.add_argument('--attack_ratio', type=float, default=0.0, help= "ratio of attacker in sampled users")
    parser.add_argument('--attack_mode', type=str, default="", choices=["sign", "noise", ""], help="type of attack")
    parser.add_argument('--aggregation', type=str, default="FedAvg", choices=["FedAvg", "atten", "Krum", "GeoMed"], help="name of aggregation method")
    parser.add_argument('--test_label_acc', action='store_true', help='obtain acc of each label and poinson acc')
    parser.add_argument('--vae_model', type=str, default="", help="directory of vae_model for detection")

    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    # sample users
    dict_users = mnist_noniid(dataset_train, args.sample_users)

    img_size = dataset_train[0][0].shape

    # build model
    input_size = 784
    num_classes = 10
    net_glob = LogisticRegression(input_size, num_classes).to(args.device)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train_epoch = []

    for iteration in range(args.epochs):
        w_locals, loss_locals = [], []
        # m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), args.sample_users, replace=False)
        print("Randomly selected {}/{} users for federated learning. {}".format(args.sample_users, args.num_users, datetime.now().strftime("%H:%M:%S")))

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # attack
        attacker_num  = int( args.attack_ratio * args.sample_users)
        attacker_idxs = np.random.choice(range(args.sample_users), attacker_num, replace=False)
        print( "{}/{} are attackers with {} attack".format(attacker_num, args.sample_users, args.attack_mode) )
        for attacker_idx in attacker_idxs:
            if args.attack_mode == "sign":
                w_locals[attacker_idx] = sign_flipping_attack(w_locals[attacker_idx])
            elif args.attack_mode == "noise":
                w_locals[attacker_idx] = additive_noise(w_locals[attacker_idx], args)
            else:
                pass

        # update global weights
        user_sizes = np.array([ len(dict_users[idx]) for idx in idxs_users ])
        user_weights = user_sizes / float(sum(user_sizes))
        if args.aggregation == "FedAvg":
            w_glob = FedAvg(w_locals, user_weights)
        else:
            w_glob = aggregation(w_locals, user_weights, args, attacker_idxs)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = np.sum(loss_locals * user_weights)

        print('=== Round {:3d}, Average loss {:.6f} ==='.format(iteration, loss_avg))
        print("{} users; time {}".format(len(idxs_users), datetime.now().strftime("%H:%M:%S")) )
        if iteration % 2 == 0:
            acc_test, loss_test = test_img(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
            print("Testing accuracy:  {:.2f}, loss: {}".format(acc_test, loss_test))
            if args.test_label_acc:
                acc_test, loss_test, acc_per_label, poison_acc = test_img_poison(copy.deepcopy(net_glob), dataset_test, args)
                print( "Testing accuracy: {:.6f} loss: {:.6}".format(acc_test, loss_test))
                print( "Testing Label Acc: {}".format(acc_per_label) )
                print( "Poison Acc: {}".format(poison_acc) )
                print( "======")

        print("Test end {}".format(datetime.now().strftime("%H:%M:%S")))

        loss_train_epoch.append(loss_avg)

    print("=== End ===")