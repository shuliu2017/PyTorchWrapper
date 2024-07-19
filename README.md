# lyl-cnn
A general framework of CNN classification tasks

- predict the yoga class

- give a score of the yoga pose

- To get the package, run the following code.
  
```
import os
import sys

try:
    if not os.path.isdir('/content/lyl-cnn'):
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

