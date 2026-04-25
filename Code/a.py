import numpy as np
from pyod_detection import load_and_preprocess_dataset, detect_outliers_pyod

data = load_and_preprocess_dataset('./Datasets/annthyroid.mat', 'annthyroid')
scores, labels, model, fit_time = detect_outliers_pyod(data, algorithm='AutoEncoder')

print("fit_time:", fit_time)
print("labels:", np.unique(labels, return_counts=True))
print("scores:", scores[:10])