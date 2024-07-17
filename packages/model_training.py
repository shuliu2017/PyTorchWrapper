import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Optional

def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Performs a single training step for the given model.

    This function trains the model on a given dataloader for one epoch and computes 
    the average loss, accuracy, precision, recall, and F1-score.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader providing the training data.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the average loss, accuracy, precision, recall, and F1-score.
    """
    model.train()
    train_loss = 0
    num_batches = len(dataloader)
    
    all_targets = []
    all_preds = []
    
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    avg_loss = train_loss / num_batches
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).sum().item() / len(all_targets)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def evaluation_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    """
    Performs a single evaluation step for the given model.

    This function evaluates the model on a given dataloader (validation or test) 
    and computes the average loss, accuracy, precision, recall, and F1-score.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (DataLoader): The DataLoader providing the validation data.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the average loss, accuracy, precision, recall, and F1-score.
    """
    model.eval()
    val_loss = 0
    num_batches = len(dataloader)
    
    all_targets = []
    all_preds = []
    
    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = val_loss / num_batches
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).sum().item() / len(all_targets)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

class EarlyStopping:
    def __init__(self, patience=8, verbose=True, delta=0, path='early_stopping_checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'Initial Validation loss ({val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          early_stopping: Optional[EarlyStopping] = None,
          writer: torch.utils.tensorboard.writer.SummaryWriter = None
          ) -> pd.DataFrame:

    train_metrics = pd.DataFrame()
    valid_metrics = pd.DataFrame()

    for epoch in tqdm(range(epochs)):
        train_metric = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        valid_metric = evaluation_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)
 
        if writer:
            for key in train_metric.keys():
                writer.add_scalars(main_tag=key,
                               tag_scalar_dict={f"{key}_train": train_metric[key],
                                                "{key}_valid": valid_metric[key]},
                               global_step=epoch +1)

            writer.close()

        train_metric['epoch'] = epoch + 1
        valid_metric['epoch'] = epoch + 1

        train_metric_df = pd.DataFrame([train_metric])
        valid_metric_df = pd.DataFrame([valid_metric])
        train_metrics = pd.concat([train_metrics, train_metric_df], ignore_index=True)
        valid_metrics = pd.concat([valid_metrics, valid_metric_df], ignore_index=True)
        
        if early_stopping is not None:
            early_stopping(valid_metric['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    results = pd.merge(train_metrics, valid_metrics, on="epoch", suffixes=['train', 'valid'])

    return results
