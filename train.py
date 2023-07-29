# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Template Project (https://github.com/victoresque/pytorch-template)
# Copyright (c) 2018 Victor Huang. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import collections
import torch
import numpy as np
import base.base_dataloader as module_dataloader
import dataset as module_dataset
import model.loss as module_loss
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, args):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_dataset = config.init_obj('train_dataset', module_dataset,
                                    graph_truncation_dist=config['graph_truncation_dist'])
    train_dataloader = config.init_obj('train_data_loader', module_dataloader, dataset=train_dataset)

    val_dataset = config.init_obj('val_dataset', module_dataset,
                                  graph_truncation_dist=config['graph_truncation_dist'])

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, num_classes=train_dataset.num_classes)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model=model,
                      criterion=criterion,
                      metric_ftns=None,
                      optimizer=optimizer,
                      config=config,
                      device=device,
                      data_loader=train_dataloader,
                      len_epoch=config['trainer']['len_epoch'] if 'len_epoch' in config['trainer'].keys() else None,
                      valid_dataset=val_dataset,
                      eval_interval=config['trainer']['eval_interval'],
                      active_track_thresh=config['trainer']['active_track_thresh'],
                      lr_scheduler=None)

    if args.eval_only:
        trainer.val(args.eval_output)
    else:
        trainer.train()
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--eval_only', action='store_true',
                      help='whether to run in eval only mode')
    args.add_argument('-o', '--eval_output', default=None, type=str,
                      help='Output files of evaluation')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--le', '--len_epoch'], type=int, target='trainer;len_epoch'),
        CustomArgs(['--n', '--name'], type=str, target='name'),
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main(config, args)
