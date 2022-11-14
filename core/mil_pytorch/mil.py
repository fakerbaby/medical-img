from torch import nn
from torch.utils.data import Dataset
import torch
import logging
import torchvision.models as models
import numpy as np

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BagModel(nn.Module):
  '''
  Model for solving MIL problems

  Args:
    prepNN: neural network created by user processing input before aggregation function (subclass of torch.nn.Module)
    afterNN: neural network created by user processing output of aggregation function and outputing final output of BagModel (subclass of torch.nn.Module)
    aggregation_func: mil.max and mil.mean supported, any aggregation function with argument 'dim' and same behaviour as torch.mean can be used

  Returns:
    Output of forward function.
  '''

  def __init__(self, prepNN, afterNN, aggregation_func):
    super().__init__()
    self.prepNN = prepNN
    self.aggregation_func = aggregation_func
    self.afterNN = afterNN

  
  
  def forward(self, input):  
    ids = input[1]
    input = input[0]
    # Modify shape of bagids if only 1d tensor
    if (len(ids.shape) == 1):
      ids.resize_(1, len(ids))

    inner_ids = ids[len(ids)-1]
    device = input.device

    NN_out = self.prepNN(input)
    # print("NN_out", NN_out.shape)

    unique, inverse, counts = torch.unique(inner_ids, sorted = True, return_inverse = True, return_counts = True)
    idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
    bags = unique[idx]
    counts = counts[idx]

    # decide which instance to pass, we assume model can decide cause we can change at late trainning stage
    #build a gate
    
    gate = nn.Sequential(
      nn.Linear(1000, 1),
      # nn.Softmax()
    ).to(device)
    NN_out_ = gate(NN_out.detach()).squeeze()
    prob_list = nn.functional.softmax(NN_out_)
    print(prob_list)
    max_prob, choose = torch.max(prob_list, 0)
    print(max_prob, choose)
    output = torch.empty((len(bags), len(NN_out[0])), device = device)

    output[0] = NN_out[choose.item()]
    # output[0] = self.aggregation_func(NN_out)
    
    # for i, bag in enumerate(bags):
    # # # #   # calibrate instances level to bags level through a preNN and aggregation function
    #   output[i], b  = self.aggregation_func(NN_out[inner_ids == bag], dim = 0)
    
    
    # print(output.shape)
    # output[i] = a
    #   # print(b)
    # print("before afternn output", output.shape)
    output = self.afterNN(output)
    # print("after afternn output", output.shape)
    # output = (output)
    if (ids.shape[0] == 1):
      return output
    else:
      ids = ids[:len(ids)-1]
      mask = torch.empty(0, device = device).long()
      for i in range(len(counts)):
        mask = torch.cat((mask, torch.sum(counts[:i], dtype = torch.int64).reshape(1)))
      return (output, ids[:,mask])


class MilDataset(Dataset):
  '''
  Subclass of torch.utils.data.Dataset. 

  Args:
    data:
    ids:
    labels:
    normalize:
  '''
  def __init__(self, data, ids, labels, normalize=True):
    self.data = data
    self.labels = labels
    self.ids = ids

    # Modify shape of bagids if only 1d tensor
    if (len(ids.shape) == 1):
      ids.resize_(1, len(ids))
  
    self.bags = torch.unique(self.ids[0])
  
    # Normalize
    if normalize:
      std = self.data.std(dim=0)
      mean = self.data.mean(dim=0)
      self.data = (self.data - mean)/std

  def __len__(self):
    return len(self.bags)
  
  def __getitem__(self, index):
    data = self.data[self.ids[0] == self.bags[index]]
    bagids = self.ids[:, self.ids[0] == self.bags[index]]
    labels = self.labels[index]

    return data, bagids, labels
  
  def n_features(self):
    return self.data.size(1)


def collate(batch):
  '''
  '''
  batch_data = []
  batch_bagids = []
  batch_labels = []
  for sample in batch:
    batch_data.append(sample[0])
    batch_bagids.append(sample[1])
    batch_labels.append(sample[2])
  
  out_data = torch.cat(batch_data, dim = 0)
  out_bagids = torch.cat(batch_bagids, dim = 1)
  out_labels = torch.stack(batch_labels)
  
  return out_data, out_bagids, out_labels


def collate_np(batch):
  '''

  '''
  batch_data = []
  batch_bagids = []
  batch_labels = []
  
  for sample in batch:
    batch_data.append(sample[0])
    batch_bagids.append(sample[1])
    batch_labels.append(sample[2])
  
  out_data = torch.cat(batch_data, dim = 0)
  out_bagids = torch.cat(batch_bagids, dim = 1)
  out_labels = torch.tensor(batch_labels)
  
  return out_data, out_bagids, out_labels



