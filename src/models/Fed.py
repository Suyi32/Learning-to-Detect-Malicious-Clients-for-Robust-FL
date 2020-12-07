#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np

def FedAvg(w, weights):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weights[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weights[i]
    return w_avg


if __name__ == '__main__':
	w = [ {"a": 1, "b": np.array([1,2])}, {"a": 2, "b": np.array([3,4])}, {"a": 3, "b": np.array([5,6])}, {"a": 4, "b": np.array([7,8])}]
	weights = [0.1, 0.2, 0.3, 0.4]
	print(FedAvg(w, weights))