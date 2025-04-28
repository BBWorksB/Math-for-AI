import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
data = scipy.io.loadmat('face_data.mat')

# Step 2: Extract the variables
images = data['image']  
personID = data['personID'].flatten()  
subsetID = data['subsetID'].flatten()

# Step 3: Access one image)g
sample_image = images[0][3]  

print(f"Sample image shape: {sample_image.shape}")  

# Step 4: Visualize the image
plt.imshow(sample_image, cmap='gray')
plt.title(f'Person ID: {personID[0]}, Subset ID: {subsetID[0]}')
plt.axis('off')
plt.show()
