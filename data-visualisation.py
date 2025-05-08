import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load a .mat file,
mat = scipy.io.loadmat('Xtrain.mat')

# Extract the time series data
time_series_data = mat['Xtrain']

# Improved Printing 
print("Keys in the .mat file:", mat.keys())  # Show the keys to understand the structure
print("Shape of Xtrain:", time_series_data.shape) # Show the shape of the data
print("Data type of Xtrain:", time_series_data.dtype) # Show the data type

# Print the first few values:
print("\nFirst 10 values of Xtrain:\n", time_series_data[:10])

# Ensure time_series_data is 1D for plotting
time_series_data_1d = time_series_data.flatten()

plt.plot(time_series_data_1d)
plt.title('Time Series Data from Xtrain.mat')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()