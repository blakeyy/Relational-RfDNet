# Training script
# author: ynie
# date: Feb, 2020
from models.optimizers import load_optimizer, load_scheduler, load_bnm_scheduler
from net_utils.utils import load_device, load_model, load_trainer, load_dataloader
from net_utils.utils import CheckpointIO
from train_epoch import train
from configs.config_utils import mount_external_config

def run(cfg):
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    train_loader = load_dataloader(cfg, mode='train')
    val_loader = load_dataloader(cfg, mode='val')

    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load optimizer'''
    cfg.log_string('Loading optimizer.')
    optimizer = load_optimizer(config=cfg.config, net=net)
    checkpoint.register_modules(optimizer=optimizer)

    '''Load scheduler'''
    cfg.log_string('Loading optimizer scheduler.')
    scheduler = load_scheduler(config=cfg.config, optimizer=optimizer)
    checkpoint.register_modules(scheduler=scheduler)

    '''Check existing checkpoint (resume or finetune)'''
    checkpoint.parse_checkpoint()

    '''BN momentum scheduler'''
    bnm_scheduler = load_bnm_scheduler(cfg=cfg, net=net, start_epoch=scheduler.last_epoch)

    '''Load trainer'''
    cfg.log_string('Loading trainer.')
    trainer = load_trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)

    '''Start to train'''
    cfg.log_string('Start to train.')
    cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))
    cfg.log_string('Total number of trainable parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters() if p.requires_grad)))

    train(cfg=cfg, trainer=trainer, scheduler=scheduler, bnm_scheduler=bnm_scheduler, checkpoint=checkpoint, train_loader=train_loader, val_loader=val_loader)

    cfg.log_string('Training finished.')