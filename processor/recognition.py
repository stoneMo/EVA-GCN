#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import time
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torch.autograd import Variable
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
import random

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        random.seed(1)
        self.model.apply(weights_init)
        #self.loss = nn.CrossEntropyLoss()
        self.r_loss =  nn.MSELoss()  
        self.c_loss = nn.CrossEntropyLoss() 
        self.mae = nn.L1Loss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.2**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
          
        else:
            self.lr = self.arg.base_lr
        

    #def show_topk(self, k):
    #    rank = self.result.argsort()
    #    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
    #    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    #    self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()             # set the net.st_gcn.Model to training mode
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        # new add
        softmax = nn.Softmax().cuda()
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda()
          
        for i, (data, label) in enumerate(loader):

            # get data
            data = data.float().to(self.dev)
            #label = label.long().to(self.dev)
            label = label.float().to(self.dev)
            pitch_label, yaw_label, roll_label = label[:, 0], label[:, 1], label[:, 2]

            # forward
            output = self.model(data)
            #if i == 2:
            #    assert(0)
            pitch, yaw, roll = output[:, 0], output[:, 1], output[:, 2]          
            
            # loss
            alpha, beta = 0.001,0.0015
            #alpha, beta = 0.001, 0.0015
            #Yaw_loss = alpha * torch.mean(self.r_loss(yaw, yaw_label), axis=0)
            Pitch_loss = beta * self.r_loss(pitch, pitch_label)
            Yaw_loss = alpha * self.r_loss(yaw, yaw_label)
            #Roll_loss = alpha * torch.mean(self.r_loss(roll, roll_label), axis=0)
            Roll_loss = alpha * self.r_loss(roll, roll_label) 
            loss = Yaw_loss + Pitch_loss + Roll_loss
            #loss = Pitch_loss


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()

            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            # self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        result_frag = []
        label_frag = []
        pose_value = []

        for data, label in loader:
            
            # get data
            data = data.float().to(self.dev)
            #label = label.long().to(self.dev)
            label = label.float().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            
            result_frag.append(output.data.cpu().numpy())
            #np.save('demo_video_gcn.npy', result_frag) 
            #assert(0)
            # get loss
            if evaluation:
                pose = np.abs(output.data.cpu().numpy()-label.data.cpu().numpy())
                pose_value.append(pose)
                label_frag.append(label.data.cpu().numpy())
        
        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            #print(np.abs(self.result-self.label))
            pose = np.mean(np.abs(self.result-self.label), axis=0)
            mae_pose = np.mean(pose)
            self.epoch_info['[Pitch, Yaw, Roll]'] = pose
            self.epoch_info['MAE of (Pitch, Yaw, Roll)'] = mae_pose
            print('[Pitch, Yaw, Roll]: ', pose)
            print('MAE of (Pitch, Yaw, Roll)', mae_pose)
            #np.save('pose.npy', self.result)
            #np.save('pose-label.npy', np.abs(self.result-self.label))
            # show top-k accuracy
           # for k in self.arg.show_topk:
           #     self.show_topk(k)
            return mae_pose
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        #parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
