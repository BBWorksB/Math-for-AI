import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

lambda_rate = 2
num_iter = 5000
m_values = range(1, 21)

sync_runtime = []
async_runtime = []

for m in m_values:
    sample = np.random.exponential(scale=1/lambda_rate, size=(num_iter, m))

    # Synchronous: max of each row
    sync_average = np.mean(np.max(sample, axis=1))
    sync_runtime.append(sync_average)

    # Assynchoronus: mean of each row
    async_average = np.mean(np.mean(sample, axis=1))
    async_runtime.append(async_average)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(m_values, sync_runtime, label='Synchronous SGD', marker='o')
plt.plot(m_values, async_runtime, label='Asynchronous SGD', marker='s')
plt.xlabel('Number of Worker Nodes (m)')
plt.ylabel('Average Runtime per Iteration')
plt.title('Synchronous vs Asynchronous SGD Runtimes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 

# For part e
# Harmonic numbers for synchronous SGD
harmonic_numbers = np.array([np.sum(1 / np.arange(1, m + 1)) for m in m_values])
expected_sync = (1 / lambda_rate) * harmonic_numbers

# Expected runtime for asynchronous SGD is constant
expected_async = np.full_like(m_values, 1 / lambda_rate, dtype=np.float64)

# Plot: Simulated vs Theoretical runtimes
plt.figure(figsize=(10, 6))
plt.plot(m_values, sync_runtime, label='Simulated Sync SGD', marker='o')
plt.plot(m_values, async_runtime, label='Simulated Async SGD', marker='s')
plt.plot(m_values, expected_sync, label='Theoretical Sync SGD', linestyle='--', color='blue')
plt.plot(m_values, expected_async, label='Theoretical Async SGD', linestyle='--', color='orange')
plt.xlabel('Number of Worker Nodes (m)')
plt.ylabel('Average Runtime per Iteration')
plt.title('Simulated vs Theoretical SGD Runtimes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()