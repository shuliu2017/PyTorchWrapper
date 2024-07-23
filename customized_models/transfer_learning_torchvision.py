import torch
from torch import nn
import torchvision

def initialize_effnetb2(out_features: int = 8,
                        dropout: float = 0.2) -> nn.Module:
    """
    Initializes an EfficientNet_B2 model with a custom classifier head.

    Args:
        out_features (int): The number of output features for the classifier head. Default is 8.
        dropout (float): The dropout rate for the classifier head. Default is 0.2.

    Returns:
        nn.Module: The EfficientNet_B2 model with the updated classifier head.
    """
                          
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
                          
    for param in model.features.parameters():
        param.requires_grad=False

    # name = 'EfficientNet_B2;
    # model.name = name
    
    in_features=1408

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,
                  out_features=out_features)
    )

    return model

def initialize_effnetv2s(out_features: int = 8,
                         dropout: float = 0.2) -> nn.Module:
    """
    Initializes an EfficientNet_V2_S model with a custom classifier head.

    Args:
        out_features (int): The number of output features for the classifier head. Default is 8.
        dropout (float): The dropout rate for the classifier head. Default is 0.2.

    Returns:
        nn.Module: The EfficientNet_V2_S model with the updated classifier head.
    """

    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,
                  out_features=out_features)
    )

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
