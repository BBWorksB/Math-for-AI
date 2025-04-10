import numpy as np

# Original data
data = np.array([
    [0, -1, -2],
    [1,  1,  1],
    [2,  0,  1]
])

# Step 0: Compute the mean
mean_vector = np.mean(data, axis=0)

# Step 1: Center the data
X_centered = data - mean_vector

# Step 2: Given covariance matrix
Cx = np.array([
    [2, 1, 3],
    [1, 2, 3],
    [3, 3, 6]
])

# Step 3: Eigen decomposition
eigvals, eigvecs = np.linalg.eig(Cx)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# Step 4: Choose top 2 principal components (eigenvectors)
U = eigvecs[:, :2]  # Take first two columns (PC1 and PC2)

# Step 5: Project centered data onto 2D principal subspace
A = X_centered @ U

# Step 6: Reconstruct the data using the top 2 PCs
X_reconstructed = A @ U.T + mean_vector

# Step 7: Compute reconstruction errors
errors = np.sum((data - X_reconstructed)**2, axis=1)

# Display results
print("\nProjection coefficients (a_i):\n", A)
print("\nReconstructed data (x_i_hat):\n", X_reconstructed)
print("\nReconstruction errors ||x_i - x_i_hat||^2:\n", errors)
