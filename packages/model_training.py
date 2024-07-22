import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from typing import Optional, Dict, Callable, Tuple

def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metrics: Optional[Dict[str, Tuple[Callable, Dict]]],
               device: torch.device
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
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        
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
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    avg_loss = train_loss / num_batches
    scores = {'loss': avg_loss}

    if metrics:
        for metric_name, (metric_fn, metric_params) in metrics.items():
            scores[metric_name] = metric_fn(all_targets, all_preds, **metric_params)
    
    return scores


def evaluation_step(model: torch.nn.Module,
                    dataloader: DataLoader,
                    loss_fn: torch.nn.Module,
                    metrics: Optional[Dict[str, Tuple[Callable, Dict]]],
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
            loss = loss_fn(outputs, targets)
            
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = val_loss / num_batches
    scores = {'loss': avg_loss}

    if metrics:
        for metric_name, (metric_fn, metric_params) in metrics.items():
            scores[metric_name] = metric_fn(all_targets, all_preds, **metric_params)
    
    return scores

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
          validation_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          metrics: Optional[dict],
          epochs: int,
          early_stopping: Optional[EarlyStopping] = None,
          device: torch.device = 'cpu',
          writer: tensorboard.writer.SummaryWriter = None
          ) -> pd.DataFrame:

    train_scores = pd.DataFrame()
    valid_scores = pd.DataFrame()

    for epoch in tqdm(range(epochs)):
        train_score = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        metrics=metrics,
                                        device=device)
      
        valid_score = evaluation_step(model=model,
                                       dataloader=validation_dataloader,
                                       loss_fn=loss_fn,
                                       metrics=metrics,
                                       device=device)
 
        if writer:
            for key in train_score.keys():
                writer.add_scalars(
                    main_tag=key,
                    tag_scalar_dict={f"{key}_train": train_score[key],
                                    "{key}_valid": valid_score[key]},
                    global_step=epoch +1
                    )

            writer.close()

        train_score['epoch'] = epoch + 1
        valid_score['epoch'] = epoch + 1

        train_score_df = pd.DataFrame([train_score])
        valid_score_df = pd.DataFrame([valid_score])
        train_scores = pd.concat([train_scores, train_score_df], ignore_index=True)
        valid_scores = pd.concat([valid_scores, valid_score_df], ignore_index=True)
        
        if early_stopping is not None:
            early_stopping(valid_score['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    results = pd.merge(train_scores, valid_scores, on="epoch", suffixes=['_train', '_valid'])

    return results
