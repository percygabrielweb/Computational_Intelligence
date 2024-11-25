# Age, Gender, and Ethnicity Prediction Using Neural Networks

This project explores the task of predicting age, gender, and ethnicity from facial images using three different neural network models: a Convolutional Neural Network (CNN), a Multi-Layer Perceptron (MLP), and a Long Short-Term Memory (LSTM) model. The aim is to compare the performance of these architectures on the task.

## Dataset

The dataset used for this project is the **Age, Gender, and Ethnicity Face Data CSV**, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv). 

Make sure to download and place the dataset in the `data` directory before running the notebook.

---

## Requirements

### Python Version
- Python **3.9** or higher is recommended for compatibility with PyTorch.

### Required Packages
Ensure the following packages are installed before running the notebook:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
```

## Hardware Recommendations
- **NVIDIA GPU**: It is recommended to have a GPU with CUDA support for faster training times.
- **CPU Usage**: If using a CPU, training may take between 1 to 4 hours, depending on the CPU's processing power.

## How to Run the Notebook

1. **Download the Dataset**: Ensure you have downloaded the dataset from [Kaggle](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv) and placed it in the `data` folder.
2. **Open the Notebook**: Launch the project notebook in a Python environment that supports Jupyter.
3. **Run the Notebook**: Execute the cells sequentially from **top to bottom** to ensure proper execution.
4. **View the Outputs**: The notebook generates several figures to visualize the results and compare the performance of the models.
