import numpy as np

# Replace 'your_file.npy' with the actual path to your .npy file
data = np.load('embeddings.npy')

# Now you can inspect the data
print(data)
print(data.shape)  # View the dimensions of the array
print(data.dtype)  # View the data type of the array elements