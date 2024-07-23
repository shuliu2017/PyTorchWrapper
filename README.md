# PyTorchWrapper
A general framework of pytorch classification and regression tasks

  
```
import os
import sys

try:
    if not os.path.isdir('/PyTorchWrapper'):
        raise FileNotFoundError
except FileNotFoundError:
    print("(◕‿◕✿) Downloading lyl-cnn from GitHub.")
    os.system(f'git clone https://github.com/shuliu2017/lyl-cnn.git')
except Exception as e:
    print(f"An unexpected error occurred: {e}")

sys.path.append('/content/lyl-cnn/packages')
```

# Dataset
link: https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset

# Exploratory Data Analysis

# Metrics

# Models

# Performance

# Discussion

