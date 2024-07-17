import torch

def accuracy(targets, preds):
  
  accuracy = (torch.tensor(preds) == torch.tensor(targets)).sum().item() / len(targets)
  return accuracy
