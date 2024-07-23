# simpleTorchWrapper

<img src="./logo.jpg" alt="simpleTorchWrapper" title="simpleTorchWrapper" width="300" />


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

## Install requirements
```
!pip install -r /content/pyTorchWrapper/requirements.txt
```

## Available Models

- efficientNetB2
- efficientNetV2S
- VIT

## Load Modules

```
import packages as pk
import customized_models as cm
```

## Model Training


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
early_stopping = pk.model_workflow.EarlyStopping(patience=8, path=f'{model_name}_early_stopping_checkpoint.pt')
metrics = pk.customized_metrics.regression_metrics
result = pk.model_workflow.train(model=model,
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

- Regression
```
loss_fn = torch.nn.MSELoss()
task_type = 'regression'
metrics = pk.customized_metrics.regression_metrics # MSE, MAE, R2; evaluated per epoch
```

- Classification
```
loss_fn = torch.nn.CrossEntropyLoss()
task_type = 'classification'
metrics = pk.customized_metrics.classification_metrics # Accuracy, Recall, Precision, F1; evaluated per epoch
```

- Commonly used optimizer
```
torch.optim.Adam(params=model.parameters(), lr=0.001)
torch.optim.SGD(params=model.parameters(), lr=0.001)
```

## Model Evaluation
test_model = cm.vit_regressor.ViTRegressor().to(device)
pk.pytorch_tools.load_model_state(test_model, target_dir='/content', model_name= f'{model_name}_early_stopping_checkpoint.pt')
test_result = pk.model_workflow.evaluation_step(test_model, test_loader, loss_fn, metrics, task_type, device)

## Example Notebooks

- regression [simple regression on random noise](https://github.com/shuliu2017/pyTorchWrapper/blob/main/notebooks/simple_regression.ipynb)

## Links

- [PyTorch](https://pytorch.org/) The official website of PyTorch.

- [Hugging Face](https://huggingface.co/) A platform with machine learning tools.

- [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) PyTorch tutorial. Some modules of the simpleTorchWrapper are inspired by the examples in this tutorial.

## Team

LYL is an independent research and development team made up of PhDs in computer science, statistics, and physics. We are dedicated to creating innovative applications and conducting cutting-edge research to simplify daily life and enhance overall well-being. With a passion for leveraging technology to develop practical solutions, we aim to make life more convenient and enjoyable.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This project is inspired by the code from other open-source projects. We would like to thank the authors of these projects for their contributions:

- [PyTorch](https://pytorch.org/)
- [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)

