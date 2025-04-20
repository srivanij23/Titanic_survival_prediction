# main.py
import pandas as pd
from src.preprocess import preprocess_data
from src.train_model import train_evalute


# Preprocess
X, y = preprocess_data('data/tested.csv')

# Train and Evaluate
model, y_pred =train_evalute(X, y)
