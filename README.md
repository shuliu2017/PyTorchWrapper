# PyTorchWrapper
A general framework of pytorch classification and regression tasks. This package is currently under development.


In Google Colab, run the following code to download `pyTorchWrapper`.

```
import os
import sys

try:
    if not os.path.isdir('/content/pyTorchWrapper'):
        raise FileNotFoundError
except FileNotFoundError:
    print("(◕‿◕✿) Downloading pyTorchWrapper from GitHub.")
    os.system(f'git clone https://github.com/shuliu2017/pyTorchWrapper.git')
except Exception as e:
    print(f"(◕‿◕✿) An unexpected error occurred: {e}")

sys.path.append('/content/pyTorchWrapper')
```

# Model Training

```
epochs = 1

pc.pytorch_tools.set_random_seed(seed=0)

model = cm.vit_regressor.ViTRegressor()
device = pk.pytorch_tools.get_device()
model = model.to(device)
model = pk.pytorch_tools.enable_multi_gpu(model)

model_name = 'vit_regressor'
loss_fn =nn.MSELoss()
task_type = 'regression' # use 'classification' for classification tasks

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
early_stopping = pc.model_training.EarlyStopping(patience=8, path=f'{model_name}_early_stopping_checkpoint.pt')
metrics = pc.customized_metrics.regression_metrics
result = pc.model_training.train(model=model,
                                  train_dataloader=train_loader,
                                  validation_dataloader=val_loader,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  metrics=metrics,
                                  task_type=task_type,
                                  epochs=epochs,
                                  early_stopping=early_stopping,
                                  device=device)
```

# Model Evaluation
