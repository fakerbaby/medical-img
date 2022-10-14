import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("", type=str, help="xxx", default=None)
# ...

class MultiInstanceLearner(nn.lig)