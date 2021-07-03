# Base Network for other networks.
# author: ynie
# date: Feb, 2020

from models.registers import MODULES, LOSSES
import torch.nn as nn

def multi_getattr(layer, attr, default=None):
    attributes = attr.split(".")
    for i in attributes:
        try:
            layer = getattr(layer, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return layer

def multi_hasattr(layer, attr):
    attributes = attr.split(".")
    hasattr_flag = True
    for i in attributes:
        if hasattr(layer, i):
            layer = getattr(layer, i)
        else:
            hasattr_flag = False
    return hasattr_flag

class BaseNetwork(nn.Module):
    '''
    Base Network Module for other networks
    '''
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg.config, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1)))

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def freeze_modules(self, cfg):
        '''
        Freeze modules in training
        '''
        if cfg.config['mode'] == 'train':
            freeze_layers = cfg.config['train']['freeze']
            for layer in freeze_layers:
                if not multi_hasattr(self, layer):
                    continue
                for param in multi_getattr(self, layer).parameters():
                    param.requires_grad = False
                cfg.log_string('The module: %s is fixed.' % (layer))

    def set_mode(self):
        '''
        Set train/eval mode for the network.
        :param phase: train or eval
        :return:
        '''
        freeze_layers = self.cfg.config['train']['freeze']
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)

    def load_weight(self, pretrained_model, selftrained_model=None):
        model_dict = self.state_dict()
        if selftrained_model is not None:
            # remove the 'module' string.
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_model.items() if
                            '.'.join(k.split('.')[1:]) in model_dict and k.split('.')[1] in ['completion', 'skip_propagation']}
            selftrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in selftrained_model.items() if
                            '.'.join(k.split('.')[1:]) in model_dict and k.split('.')[1] in ['backbone', 'voting', 'detection']}
            #for k, v in pretrained_dict.items():
            #    print("++++ Pretrained_dict: " + str(k))
            #for k, v in selftrained_dict.items():
            #    print("++++ selftrained_dict: " + str(k))
            #print("++++ model_dict: " + str(model_dict))
            self.cfg.log_string(
                str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
            self.cfg.log_string(
                str(set([key.split('.')[0] for key in model_dict if key not in selftrained_dict])) + ' subnet missed.')
            model_dict.update(pretrained_dict)
            model_dict.update(selftrained_dict)
        else:
            # remove the 'module' string.
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_model.items() if
                            '.'.join(k.split('.')[1:]) in model_dict}
            #for k, v in pretrained_dict.items():
            #    print("++++ Pretrained_dict: " + str(k))
            self.cfg.log_string(
                str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
            model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_optim_spec(self, config, net_spec):
        # load specific optimizer parameters
        if config['mode'] == 'train':
            if 'optimizer' in net_spec.keys():
                optim_spec = net_spec['optimizer']
            else:
                optim_spec = config['optimizer']  # else load default optimizer
        else:
            optim_spec = None

        return optim_spec

    def forward(self, *args, **kwargs):
        ''' Performs a forward step.
        '''
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        ''' calculate losses.
        '''
        raise NotImplementedError