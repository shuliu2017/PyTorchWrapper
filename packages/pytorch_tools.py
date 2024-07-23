import os
from datetime import datetime
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def set_random_seed(seed: int=0):
    """Set random seed for numpy, torch, and torch.cuda.

    Args:
          seed (int: Random seed. Default is 0.
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device() -> torch.device:
    """
    Returns the available device (CPU or CUDA).

    Returns:
        torch.device: The device to be used for computations.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def enable_multi_gpu(model: torch.nn.Module, verbose: bool = True) -> torch.nn.Module:
    """
    Configures the model to use multiple GPUs if available.

    This function checks the number of available GPUs and, if more than one GPU is available,
    it wraps the model with `nn.DataParallel` to enable data parallelism. If no GPUs are available,
    it uses the CPU. Optionally, it prints the status of the device configuration.

    Args:
        model (torch.nn.Module): The model to be configured for multi-GPU usage.
        verbose (bool, optional): If True, prints the device configuration status. Default is True.

    Returns:
        torch.nn.Module: The configured model, either wrapped with `nn.DataParallel` for multi-GPU
        usage or unchanged for single GPU or CPU usage.
    """
    if torch.cuda.device_count() == 0:
        if verbose:
            print('(◕‿◕✿) Using CPU.')
        return model
    else:
        if verbose:
            print(f"(◕‿◕✿) Using {torch.cuda.device_count()} GPUs.")
        if torch.cuda.device_count() > 1:
            return torch.nn.DataParallel(model)
        else:
            return model

def create_image_classification_dataloader(data_dir: str, transform: transforms = transforms.ToTensor, 
                      batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Creates a DataLoader for image classification tasks.

    Args:
        data_dir (str): The directory where the image data is stored.
        transform (transforms, optional): Transformations to apply to the images. Defaults to transforms.ToTensor.
        batch_size (int, optional): Number of samples per batch to load. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 0.

    Returns:
        DataLoader: A DataLoader object for the dataset.
        list: A list of class names corresponding to the folders in the dataset directory.
    """
    data = datasets.ImageFolder(data_dir, transform=transform)
    class_names = data.classes
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, class_names

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

def save_model_state(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves the PyTorch model's state dictionary.

    Args:
        model (torch.nn.Module): The model to be saved.
        target_dir (str): The directory path where the model will be saved.
        model_name (str): The filename for the saved model (should end with .pth or .pt).

    Returns:
        None
    """
    # Ensure the directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Ensure the model_name ends with .pth or .pt
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "(◕‿◕✿) model_name should end with .pth or .pt"

    # Create the full path
    full_path = os.path.join(target_dir, model_name)

    # Save the model state dictionary
    torch.save(obj=model.state_dict(), f=full_path)

    print(f"(◕‿◕✿) Model state saved to {full_path}")



def load_model_state(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Loads the PyTorch model's state dictionary.

    Args:
        model (torch.nn.Module): The model to load the state dictionary into.
        target_dir (str): The directory path where the model is saved.
        model_name (str): The filename for the saved model (should end with .pth or .pt).

    Returns:
        None
    """
    # Ensure the model_name ends with .pth or .pt
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "(◕‿◕✿) model_name should end with .pth or .pt"

    # Create the full path
    full_path = os.path.join(target_dir, model_name)

    # Load the state dictionary
    model.load_state_dict(torch.load(full_path))

    print(f"Model state loaded from {full_path}")
    
