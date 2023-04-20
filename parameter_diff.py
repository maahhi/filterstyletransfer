import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

csv_file_path = './model/styletransfer/eq/fparam-a.csv'
data = pd.read_csv(csv_file_path)
actual_values = data.iloc[:, 1:7]
estimated_values = data.iloc[:, 7:12]

pca = PCA(n_components=1)
actual_pca = pca.fit_transform(actual_values)
estimated_pca = pca.fit_transform(estimated_values)

mse = np.mean((actual_pca - estimated_pca) ** 2)
print(f'Total Loss (MSE): {mse}')

window_size = 5  # Choose an appropriate window size for the moving average
actual_pca_moving_avg = moving_average(actual_pca.flatten(), window_size)
estimated_pca_moving_avg = moving_average(estimated_pca.flatten(), window_size)

plt.figure(figsize=(12, 6))
plt.plot(actual_pca_moving_avg, label='Actual PCA (Moving Average)')
plt.plot(estimated_pca_moving_avg, label='Estimated PCA (Moving Average)')
plt.xlabel('Index')
plt.ylabel('PCA Value')
plt.legend()
plt.title('Actual vs. Estimated PCA Values with Moving Average')
plt.savefig('pca_moving_average_plot.png')
plt.show()
