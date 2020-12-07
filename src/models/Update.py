#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from models.test import test_img_poison
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        if args.dataset == 'femnist':
            self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdatePoison_MNIST(object):
    # used for MNIST experiments
    def __init__(self, args, dataset=None, idxs=None, user_idx=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.dataset = dataset
        
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.user_idx = user_idx

        self.poison_attack = False

        self.attacker_flag = False
    def train(self, net):
        net.train()
        # train and update
        shared_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.attack_mode == "poison":
                    for label_idx in range(len(labels)):
                        if labels[label_idx] == 7:
                            self.poison_attack = True
                            self.attacker_flag = True
                            labels[label_idx] = 5
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # model replacement
        trained_weights = copy.deepcopy(net.state_dict())
        scale_up = 5
        if self.args.attack_mode == "poison" and self.poison_attack:
            # print( "Scale up: {} for non-iid MNIST training".format(scale_up) )
            # print( "Poisoning test acc", test_img_poison(copy.deepcopy(net), self.dataset, self.args) )
            # print("Poisoning weights...")
            attack_weights = copy.deepcopy(shared_weights)
            for key in shared_weights.keys():
                difference =  trained_weights[key] - shared_weights[key]
                attack_weights[key] += scale_up * difference

            return attack_weights, sum(epoch_loss) / len(epoch_loss), self.attacker_flag

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.attacker_flag

class LocalUpdatePoison(object):
    # used for FEMNIST experiments
    def __init__(self, args, dataset=None, idxs=None, user_idx=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.dataset = dataset

        if args.dataset == 'femnist':
            self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.user_idx = user_idx

    def train(self, net):
        net.train()
        # train and update
        shared_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                # data poison
                if self.user_idx in self.args.attacker_idxs and self.args.attack_mode == "poison":
                    for label_idx in range(len(labels)):
                        if labels[label_idx] == 7:
                            labels[label_idx] = 5
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # model replacement
        trained_weights = copy.deepcopy(net.state_dict())
        scale_up = self.args.sample_users / float(len(self.args.attacker_idxs))
        if self.user_idx in self.args.attacker_idxs and self.args.attack_mode == "poison":
            print( "Scale up: {} = {}/{}, user: {}".format(scale_up, self.args.sample_users, len(self.args.attacker_idxs), self.user_idx) )
            # print( "Poisoning test acc", test_img_poison(copy.deepcopy(net), self.dataset, self.args) )
            # print("Poisoning weights...")
            attack_weights = copy.deepcopy(shared_weights)
            for key in shared_weights.keys():
                difference =  trained_weights[key] - shared_weights[key]
                attack_weights[key] += scale_up * difference

            return attack_weights, sum(epoch_loss) / len(epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
