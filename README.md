# simpleTorchWrapper

<img src="./logo.jpg" alt="simpleTorchWrapper" title="simpleTorchWrapper" width="500" />


A general framework of pytorch classification and regression tasks. This package is currently under development.

This package is lite and simple to use.

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

# install requirements
```
!pip install -r /content/pyTorchWrapper/requirements.txt
```

# Load Modules

```
import packages as pk
import customized_models as cm
```

# Model Training


- Regression

```
# configure multiprocessing
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

epochs = 1

pk.pytorch_tools.set_random_seed(seed=0)

model = cm.vit_regressor.ViTRegressor()
device = pk.pytorch_tools.get_device()
model = model.to(device)
model = pk.pytorch_tools.enable_multi_gpu(model)

model_name = 'vit_regressor'
loss_fn = torch.nn.MSELoss()
task_type = 'regression'

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
early_stopping = pk.model_training.EarlyStopping(patience=8, path=f'{model_name}_early_stopping_checkpoint.pt')
metrics = pk.customized_metrics.regression_metrics
result = pk.model_training.train(model=model,
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

- Classification
```
loss_fn = torch.nn.CrossEntropyLoss()
task_type = 'classification'
metrics = pk.customized_metrics.classification_metrics
```

# Model Evaluation

# Example Notebooks

- regression [simple regression on random noise](https://github.com/shuliu2017/pyTorchWrapper/blob/main/notebooks/simple_regression.ipynb)
