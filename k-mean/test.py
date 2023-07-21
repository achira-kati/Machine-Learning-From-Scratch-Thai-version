import matplotlib.pyplot as plt
import numpy as np
from k_mean import KMean

X = np.load('F:/Documents/ML/k-mean/data/X.npy')

model = KMean()
model.fit(X)