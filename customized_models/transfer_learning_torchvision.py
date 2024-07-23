import torch
from torch import nn
import torchvision
import pytorch_tools

def initialize_effnetb2(out_features: int = 8,
                        dropout: float = 0.2,
                        device: torch.device = torch.device('cpu'),
                        random_seed: int = 0) -> nn.Module:
    """
    Initializes an EfficientNet_B2 model with a custom classifier head.

    Args:
        out_features (int): The number of output features for the classifier head. Default is 8.
        dropout (float): The dropout rate for the classifier head. Default is 0.2.
        device (torch.device): The device to run the model on. Default is 'cpu'.
        random_seed (int): The random seed for reproducibility. Default is 0.

    Returns:
        nn.Module: The EfficientNet_B2 model with the updated classifier head.
    """

    pytorch_tools.set_random_seed(random_seed)

    device = pytorch_tools.get_device()
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    pytorch_tools.freeze_base_layers(model)

    # name = 'EfficientNet_B2;
    # model.name = name
    
    in_features=1408

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,
                  out_features=out_features)
    )

    model = model.to(device)

    return model

def initialize_effnetv2s(out_features: int = 8,
                         dropout: float = 0.2,
                         device: torch.device = torch.device('cpu'),
                         random_seed: int = 0) -> nn.Module:
    """
    Initializes an EfficientNet_V2_S model with a custom classifier head.

    Args:
        out_features (int): The number of output features for the classifier head. Default is 8.
        dropout (float): The dropout rate for the classifier head. Default is 0.2.
        device (torch.device): The device to run the model on. Default is 'cpu'.
        random_seed (int): The random seed for reproducibility. Default is 0.

    Returns:
        nn.Module: The EfficientNet_V2_S model with the updated classifier head.
    """

    pytorch_tools.set_random_seed(random_seed)

    device = pytorch_tools.get_device()
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
    pytorch_tools.freeze_base_layers(model)

    in_features=1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,
                  out_features=out_features)
    )

    model = model.to(device)

    return model

def initialize_model(model_name, out_features):
    """
    Available model names: EfficientNet_B2, EfficientNet_V2_S

    Args:
        model_name: (str): The name of the model for initialization.
        out_features (int): The number of output features. 
    
    Returns:
        nn.Module: The EfficientNet_V2_S model with the updated classifier head.
    """
    if model_name == "EfficientNet_B2":
      model = initialize_effnetb2(
          out_features=out_features
      )
    elif model_name == "EfficientNet_V2_S":
        model = initialize_effnetv2s(
            out_features=out_features
        )
    else:
        raise ValueError("Invalid model name.")

    return model
