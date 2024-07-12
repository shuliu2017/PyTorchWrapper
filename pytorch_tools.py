import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def get_device() -> torch.device:
    """
    Returns the available device (CPU or CUDA).

    Returns:
        torch.device: The device to be used for computations.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def set_random_seed(seed: int=0):
    """Set random seed for numpy, torch, and torch.cuda.

    Args:
          seed (int: Random seed. Default is 0.
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def freeze_base_layers(model: torch.nn.Module):
    """
    Freezes the base layers of a given model.

    Args:
        model (torch.nn.Module): The model whose base layers are to be frozen.
    """

    for param in model.features.parameters():
        param.requires_grad=False


def create_writer(experiment_name: str, model_name: str, extra: str = '') -> SummaryWriter:
    """
    Creates a SummaryWriter object for logging training information to TensorBoard.

    This function initializes a SummaryWriter object with a directory structure that includes
    the current date, experiment name, model name, and optional additional information.

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model being trained.
        extra (str, optional): Additional information to include in the log directory. Default is None.

    Returns:
        SummaryWriter: A SummaryWriter object for logging to TensorBoard.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)

