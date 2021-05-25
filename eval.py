#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import sys
import time

time_start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(BASE_DIR)  
sys.path.append(os.path.dirname(BASE_DIR))  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40, ScanobjectNN
from model import DGCNN_cls
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, time_record
import sklearn.metrics as metrics


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp eval.py checkpoints'+'/'+args.exp_name+'/'+'eval.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def test(args, io):
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(args, partition='test', num_points=args.num_points), num_workers=8,batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    elif args.dataset == 'scanobject':
        test_loader = DataLoader(ScanobjectNN(args, partition='test', num_points=args.num_points), num_workers=8,batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()

    # for model trained on 2 gpus
    for k, v in state_dict.items():
        name=k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model.eval()

    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0,2,1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp_dgcnn_cat_modelnet', metavar='N',help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',choices=['modelnet40','scanobject'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',choices=['cos', 'step'],help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')

    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='model.t7', metavar='N',help='Pretrained model path')

    parser.add_argument('--test_rot_perturbation', type=bool, default=True, help='Rotation augmentation around 3 axis.')
    parser.add_argument('--translation', type=bool, default=False, help='Translation augmentation.')
    parser.add_argument('--jitter', type=bool, default=False, help='jitter augmentation.')

    parser.add_argument('--fa', type=bool, default = True, help='Using fa module.')
    parser.add_argument('--rtit', type=str, default='cat')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    test(args, io)
