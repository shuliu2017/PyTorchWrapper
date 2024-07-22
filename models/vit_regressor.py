import torch.nn as nn
from transformers import ViTForImageClassification

class ViTRegressor(nn.Module):
    def __init__(self):
        # super(ViTRegressor, self).__init__()
        super().__init__()
        modelVit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.base_model = modelVit
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.classifier = None
        
        self.fc = nn.Linear(self.base_model.config.hidden_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.regressor = nn.Linear(256, 1)

    def forward(self, x):
        x = self.base_model.vit(x).last_hidden_state[:, 0]
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.regressor(x)
        return x
