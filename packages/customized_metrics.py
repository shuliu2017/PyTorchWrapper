from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

classification_metrics = {
    'accuracy': (accuracy_score, {}),
    'precision': (precision_score, {'average': 'weighted'}),
    'recall': (recall_score, {'average': 'weighted'}),
    'f1_score': (f1_score, {'average': 'weighted'})
}

regression_metrics = {
    'mean_squared_error': (mean_squared_error, {})
}

"""
import torch

def accuracy(targets, preds):
  
  accuracy = (torch.tensor(preds) == torch.tensor(targets)).sum().item() / len(targets)
  return accuracy
"""
