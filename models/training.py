# Base trainer for methods.
# author: ynie
# date: Feb, 2020
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import datetime
import os
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
class BaseTrainer(object):
    '''
    Base trainer for all networks.
    '''
    def __init__(self, cfg, net, optimizer, device=None):
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.device = device

    def show_lr(self):
        '''
        display current learning rates
        :return:
        '''
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        self.cfg.log_string('Current learning rates are: ' + str(lrs) + '.')

    def get_lr(self):
        '''
        return current learning rates
        :return:
        '''
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        return lrs

    def train_step(self, data, plot_gradient, epoch):
        '''
        performs a step training
        :param data (dict): data dictionary
        :return:
        '''
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        if loss['total'].requires_grad:
            loss['total'].backward()
            if plot_gradient: 
                self.plot_grad_flow(self.net.named_parameters(), epoch)
            self.optimizer.step()

        loss['total'] = loss['total'].item()
        return loss

    def eval_loss_parser(self, loss_recorder):
        '''
        get the eval
        :param loss_recorder: loss recorder for all losses.
        :return:
        '''
        return loss_recorder['total'].avg

    def compute_loss(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize_step(self, *args, **kwargs):
        ''' Performs a visualization step.
        '''
        raise NotImplementedError

    def plot_grad_flow(self,named_parameters, epoch):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
    
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                if p.grad is not None:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
                else:
                    print(n)
        figure = plt.figure(figsize = (26.5,14.5))
        ax = figure.add_subplot(111)
        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")

        ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        ax.set_xticks(range(0,len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation="vertical")

        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        ax.set_xlabel("Layers")
        ax.set_ylabel("average gradient")
        ax.set_title("Gradient flow")
        ax.grid(True)
        ax.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        figure.tight_layout()
        plot_dir = os.path.join('plot_gradient', timestamp)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, 'gradient_'+'epoch_' + str(epoch+1) + '.png')
        figure.savefig(plot_path)
        plt.close(figure)