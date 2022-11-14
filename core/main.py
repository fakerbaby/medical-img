# Copyright 2022.10 weishen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pytorch_lightning as pl

import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from argparse import ArgumentParser

from data.data__interface import DInterface
from model.model_interface import MInterface
from utils import load_model_path_by_args
from pytorch_lightning.loggers.wandb import WandbLogger


def load_callbacks(patience=10, save_top_k=None):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val/acc',
        mode='max',
        patience=patience,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val/acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))
    #if you want to add callbacks

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    
    if load_path is not None:
        model = MInterface(**vars(args))
        # args.default_root_dir = 
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path
    logger = TensorBoardLogger(save_dir='lightning_logs', name=args.log_dir)
    #wandb logger
    if args.report_to == "wandb":
        logger = WandbLogger(project="lancet", name=args.log_dir)
    #local tensorboard logger
    
    args.callbacks = load_callbacks(patience=args.patience)
    args.logger = logger
    trainer = Trainer(logger=logger).from_argparse_args(args)
    trainer.fit(model, data_module)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    #Basic Traning Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=48, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    
    # LR Scheduler
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'AdamW'], type=str)
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine', 'linear'], type=str)
    parser.add_argument('--lr_decay_steps', default=250, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-7, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=str)

    # Training Info
    parser.add_argument('--dataset', default='img_data', type=str)
    parser.add_argument('--data_dir', default='../dataset', type=str)
    parser.add_argument('--model_name', default='baseline_net', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--gamma', default=2, type=float)
    parser.add_argument('--patience', default=5, type=int)
    
    # Model Hyperparameters
    # parser.add_argument('--hid', default=64, type=int)
    # parser.add_argument('--block_num', default=8, type=int)
    # parser.add_argument('--in_channel', default=3, type=int)
    # parser.add_argument('--layer_num', default=5, type=int)

    # Other
    # parser.add_argument("--TASK_TYPE", default="base", type=str)
    # parser.add_argument('--aug_prob', default=0.5, type=float)
    parser.add_argument('--report_to', default="none", type=str)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]
    print("used args:", args)
    main(args)