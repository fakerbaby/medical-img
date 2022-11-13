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


import inspect
import torch
import importlib
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.preds, self.target = [], []
        self.classification = {}
        
        
    def forward(self, img):
        return self.model(img)

    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    
    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        # print(logits)
        loss = self.loss_function(logits, y)
        # preds = logits > 0.5
        return loss, logits.squeeze(), y.to(torch.int).squeeze()
    
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        # print(preds)
         # update and log metrics 
        preds = preds > 0.5 
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}

    
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        
        # update and log metrics
        preds = preds > 0.5 
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        # recall 
        list1 = preds.tolist() 
        list2 = targets.tolist()
        print(list1, list2)
        self.preds.extend(list1)
        self.target.extend(list2)

        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

   
    def training_epoch_end(self, outputs):
        pass
     
     
    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        
        target_names = ['negative', 'positive']
        self.classification["test"] = classification_report(self.preds, self.target, target_names=target_names, digits=5)
        print("test/report:", self.classification["test"])



    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
        elif self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=weight_decay
            )
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False
            )

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            elif self.hparams.lr_scheduler == 'linear':
                scheduler = lrs.LinearLR(optimizer)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]


    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'focal':
            self.loss_function = FocalLoss(alpha=self.hparams.alpha, gamma=self.hparams.gamma)
        else:
            raise ValueError("Invalid Loss Type!")


    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss