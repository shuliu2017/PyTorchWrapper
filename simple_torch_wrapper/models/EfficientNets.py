from torch import nn
import torchvision
from typing import Optional

def initialize_efficinet_b0(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=1280

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b1(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    model = torchvision.models.efficientnet_b1(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=1280

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b2(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=1408

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b3(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    model = torchvision.models.efficientnet_b3(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=1536

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b4(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    model = torchvision.models.efficientnet_b4(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=1792

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b5(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
    model = torchvision.models.efficientnet_b5(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=2048

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b6(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT
    model = torchvision.models.efficientnet_b6(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=2304

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model

def initialize_efficinet_b7(out_features: int = 8,
                        classifier: Optional[nn.Sequential] = None) -> nn.Module:

    weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
    model = torchvision.models.efficientnet_b7(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False

    in_features=2560

    if classifier is None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=out_features)
            )
    else:
        model.classifier = classifier

    return model




