import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from typing import Optional, Dict, Callable, Tuple
import os

def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metrics: Optional[Dict[str, Tuple[Callable, Dict]]],
               task_type: str,
               device: torch.device,
               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
               ):
    """
    Performs a single training step for the given model.

    This function trains the model on a given dataloader for one epoch and computes 
    the average loss, accuracy, precision, recall, and F1-score.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader providing the training data.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        metrics (Optional[Dict[str, Tuple[Callable, dict]]]): A dictionary containing the metric names,
            functions, and parameters. The functions should accept two arguments: true labels and predictions.
            e.g. classification_metrics = {
                'accuracy': (accuracy_score, {}),
                'precision': (precision_score, {'average': 'weighted'})}.
        task_type (str): The type of task, either 'classification' or 'regression'.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The learning rate scheduler (default: None).
        
    Returns:
        Dict: A dictionary containing the average loss, accuracy, precision, recall, and F1-score.
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
      
        if task_type == 'classification':
            loss = loss_fn(outputs, targets)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        elif task_type == 'regression':
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, targets)
            preds = outputs.detach().cpu().numpy()
        else:
            raise ValueError("(◕‿◕✿) task_type must be either 'classification' or 'regression'")
      
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds)
    
    avg_loss = train_loss / num_batches
    scores = {'avg_batch_loss': avg_loss}

    if metrics:
        for metric_name, (metric_fn, metric_params) in metrics.items():
            scores[metric_name] = metric_fn(all_targets, all_preds, **metric_params)

    if scheduler is not None:
        scheduler.step(avg_loss)
    
    return scores


def evaluation_step(model: torch.nn.Module,
                    dataloader: DataLoader,
                    loss_fn: torch.nn.Module,
                    metrics: Optional[Dict[str, Tuple[Callable, Dict]]],
                    task_type: str,
                    device: torch.device
                   ):
    """
    Performs a single evaluation step for the given model.

    This function evaluates the model on a given dataloader (validation or test) 
    and computes the average loss, accuracy, precision, recall, and F1-score.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (DataLoader): The DataLoader providing the validation data.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        metrics (Optional[Dict[str, Tuple[Callable, dict]]]): A dictionary containing the metric names,
            functions, and parameters. The functions should accept two arguments: true labels and predictions.
            e.g. classification_metrics = {
                'accuracy': (accuracy_score, {}),
                'precision': (precision_score, {'average': 'weighted'})}.
        task_type (str): The type of task, either 'classification' or 'regression'.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        
    Returns:
        Dict: A dictionary containing the average loss, accuracy, precision, recall, and F1-score.
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
          
            if task_type == 'classification':
                loss = loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            elif task_type == 'regression':
                outputs = outputs.squeeze()
                loss = loss_fn(outputs, targets)
                preds = outputs.detach().cpu().numpy()
            else:
                raise ValueError("(◕‿◕✿) task_type must be either 'classification' or 'regression'")
            
            val_loss += loss.item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)
    
    avg_loss = val_loss / num_batches
    scores = {'avg_batch_loss': avg_loss}

    if metrics:
        for metric_name, (metric_fn, metric_params) in metrics.items():
            scores[metric_name] = metric_fn(all_targets, all_preds, **metric_params)
    
    return scores

class EarlyStopping:
    def __init__(self, patience=8, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, epoch):

        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'(◕‿◕✿) Epoch {epoch}: Initial Validation loss ({val_loss:.6f}).')
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'(◕‿◕✿) Epoch {epoch}: EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'(◕‿◕✿) Epoch {epoch}: Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).')
            self.best_loss = val_loss
            self.counter = 0

def _add_suffix_to_basename(path, suffix):
    # Get the directory name and base name
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    
    # Split the base name into name and extension
    name, ext = os.path.splitext(basename)
    
    # Add the suffix to the name
    new_basename = f"{name}{suffix}{ext}"
    
    # Reconstruct the full path with the new base name
    new_path = os.path.join(dirname, new_basename)
    
    return new_path

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          validation_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          metrics: Optional[dict],
          task_type: str,
          epochs: int,
          early_stopping: Optional[EarlyStopping] = None,
          save_freq: int = 0,
          save_path: str = 'model_checkpoint.pt',
          overwrite=True,
          device: torch.device = 'cpu',
          writer: tensorboard.writer.SummaryWriter = None
          ) -> pd.DataFrame:

    train_scores = pd.DataFrame()
    valid_scores = pd.DataFrame()

    for epoch in tqdm(range(1, epochs+1)):
        train_score = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        metrics=metrics,
                                        task_type=task_type,
                                        device=device)
      
        valid_score = evaluation_step(model=model,
                                       dataloader=validation_dataloader,
                                       loss_fn=loss_fn,
                                       metrics=metrics,
                                       task_type=task_type,
                                       device=device)
 
        if writer:
            for key in train_score.keys():
                writer.add_scalars(
                    main_tag=key,
                    tag_scalar_dict={f"{key}_train": train_score[key],
                                     f"{key}_valid": valid_score[key]},
                    global_step=epoch
                    )

            

        train_score['epoch'] = epoch
        valid_score['epoch'] = epoch

        train_score_df = pd.DataFrame([train_score])
        valid_score_df = pd.DataFrame([valid_score])
        train_scores = pd.concat([train_scores, train_score_df], ignore_index=True)
        valid_scores = pd.concat([valid_scores, valid_score_df], ignore_index=True)
        
        if early_stopping is not None:
            early_stopping(valid_score['avg_batch_loss'], model, epoch)
            if early_stopping.early_stop:
                early_stopping_path = _add_suffix_to_basename(save_path, '_early_stopping')
                torch.save(model.state_dict(), early_stopping_path)
                print(f'    (◕‿◕✿) Epoch {epoch}: Early stopping triggered. Save model to {early_stopping_path}.')
                break

        # Save the model at specified intervals
        # epoch is counted from 1
        if save_freq > 0 and (epoch-1) % save_freq == 0:
            if overwrite:
                torch.save(model.state_dict(), save_path)
                print(f"    (◕‿◕✿) Epoch {epoch}: Save model to {save_path}.")
            else:
                epoch_path = _add_suffix_to_basename(save_path, f'_{epoch}')
                torch.save(model.state_dict(), epoch_path)
                print(f"    (◕‿◕✿) Epoch {epoch}: Save model to {epoch_path}.")

    if writer:
        writer.close()

    results = pd.merge(train_scores, valid_scores, on="epoch", suffixes=['_train', '_valid'])

    return results
