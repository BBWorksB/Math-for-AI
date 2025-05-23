import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from scipy import linalg
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

class HiddenMarkovModel:
    def __init__(self, num_states, num_observations):
        """
        Initializes the HMM with random transition, emission, and initial state probabilities.
        
        Args:
            num_states (int): Number of hidden states.
            num_observations (int): Number of unique observations.
        """
        self.num_states = num_states
        self.num_observations = num_observations
        self.A = np.random.dirichlet(np.ones(num_states), num_states)  # Transition matrix
        self.B = np.random.dirichlet(np.ones(num_observations), num_states)  # Emission matrix
        self.pi = np.random.dirichlet(np.ones(num_states))  # Initial state distribution
        
        # Log-space versions to avoid underflow
        self.A_log = np.log(self.A + 1e-10)
        self.B_log = np.log(self.B + 1e-10)
        self.pi_log = np.log(self.pi + 1e-10)

    def forward_algorithm_log(self, O):
        """
        Forward algorithm in log-space.
        
        Args:
            O (np.array): Observation sequence (integers).
        
        Returns:
            alpha_log (np.array): Log-probability matrix of forward probabilities.
        
        TODO:
        - Implement the forward algorithm initialization and recursion in log-space.
        """
        T = len(O)
        N = self.num_states
        alpha_log = np.zeros((T, N))

        # TODO: Initialization step
        alpha_log[0] = self.pi_log + self.B_log[:, O[0]]  # Replace with initialization logic: log(α_0) = log(π) + log(B[:, O_0])

        # TODO: Recursion step
        for t in range(1, T):
            for j in range(N):
                alpha_log[t, j] = np.logaddexp.reduce(alpha_log[t - 1] + self.A_log[:, j]) + self.B_log[j, O[t]]  # Replace with recursion logic

        return alpha_log

    def backward_algorithm_log(self, O):
        """
        Backward algorithm in log-space.
        
        Args:
            O (np.array): Observation sequence (integers).
        
        Returns:
            beta_log (np.array): Log-probability matrix of backward probabilities.
        
        TODO:
        - Implement the backward algorithm initialization and recursion in log-space.
        """
        T = len(O)
        N = self.num_states
        beta_log = np.zeros((T, N))

        # TODO: Initialization step
        beta_log[-1] = 0  # log(1) = 0

        # TODO: Recursion step
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta_log[t, i] = np.logaddexp.reduce(self.A_log[i, :] + self.B_log[:, O[t + 1]] + beta_log[t + 1])  # Replace with recursion logic

        return beta_log

    def baum_welch_log(self, O, max_iter=100, epsilon=1e-6):
        """
        Baum-Welch algorithm for training HMM in log-space.
        
        Args:
            O (np.array): Observation sequence (integers).
            max_iter (int): Maximum number of iterations.
            epsilon (float): Small value to prevent division by zero.
        
        Returns:
            Trained transition, emission, and initial state distributions.
        
        TODO:
        - Implement the update steps for transition and emission probabilities.
        """
        T = len(O)
        
        for iteration in range(max_iter):
            # TODO: Call forward_algorithm_log and backward_algorithm_log
            alpha_log = self.forward_algorithm_log(O)  # Call forward algorithm
            beta_log = self.backward_algorithm_log(O)  # Call backward algorithm

            # Compute gamma and xi in log-space
            gamma_log = alpha_log + beta_log - np.logaddexp.reduce(alpha_log[-1])
            xi_log = np.zeros((T - 1, self.num_states, self.num_states))

            for t in range(T - 1):
                denom_log = np.logaddexp.reduce(
                                                    alpha_log[t, :, None] + self.A_log + self.B_log[:, O[t + 1]] + beta_log[t + 1],
                                                    axis=(0, 1),
                                                )  # Replace with logic for computing the denominator
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        xi_log[t, i, j] = alpha_log[t, i] + self.A_log[i, j] + self.B_log[j, O[t + 1]] + beta_log[t + 1, j] - denom_log  # Replace with logic for computing xi_log

            # TODO: Update A_log, B_log, and pi_log
            self.A_log = np.logaddexp.reduce(xi_log, axis=0) - np.logaddexp.reduce(gamma_log[:-1], axis=0)[:, None]  # Replace with logic for updating A_log
            self.B_log = np.zeros_like(self.B_log)  # Replace with logic for updating B_log
            for k in range(self.num_observations):
                mask = (O == k)
                self.B_log[:, k] = np.logaddexp.reduce(gamma_log[mask], axis=0) - np.logaddexp.reduce(gamma_log, axis=0)
            self.pi_log = gamma_log[0]
            # self.pi_log = NotImplemented  # Replace with logic for updating pi_log

        return np.exp(self.A_log), np.exp(self.B_log), np.exp(self.pi_log)

    def viterbi_algorithm_log(self, O):
        """
        Viterbi algorithm for finding the most likely state sequence in log-space.
        
        Args:
            O (np.array): Observation sequence (integers).
        
        Returns:
            states (np.array): Most likely state sequence.
        
        TODO:
        - Implement the Viterbi algorithm's initialization and recursion steps.
        """
        T = len(O)
        N = self.num_states
        delta_log = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # TODO: Initialization step
        delta_log[0] = self.pi_log + self.B_log[:, O[0]]  # Replace with initialization logic

        # TODO: Recursion step
        for t in range(1, T):
            for j in range(N):
                delta_log[t, j] = np.max(delta_log[t - 1] + self.A_log[:, j]) + self.B_log[j, O[t]]  # Replace with recursion logic
                psi[t, j] = np.argmax(delta_log[t - 1] + self.A_log[:, j])  # Replace with logic for tracking the path

        # TODO: Path backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta_log[-1])  # Replace with logic for the final state
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]  # Replace with path backtracking logic

        return states
    
    import numpy as np



# Main function with TODOs for students to complete
if __name__ == "__main__":
    # Load and preprocess the weather dataset
    weather_df = pd.read_csv('seattle-weather.csv', parse_dates=True, index_col=0) # TODO Load the data using the 'parse_dates' and 'index_col' arguments
    weather_df = weather_df[['temp_max', 'temp_min', 'weather']].dropna()  # TODO: Select 'temp_max', 'temp_min', and 'weather' columns and drop NaN values
  

    # Calculate the average temperature and encode the 'weather' column
    weather_df['temp_avg'] = (weather_df['temp_max'] + weather_df['temp_min']) / 2 #TODO Calculate the average temperature

    weather_mapping = {label: idx for idx, label in enumerate(weather_df['weather'].unique())}
    reverse_weather_mapping = {v: k for k, v in weather_mapping.items()}
    weather_df['weather_encoded'] = weather_df['weather'].map(weather_mapping)

    # Convert average temperature to integer values for the observation sequence
    temp_min = weather_df['temp_avg'].min()
    O = (weather_df['temp_avg'] - temp_min).astype(int).values  # Observation sequence

    # Split data into training and testing sets
    train_size = int(0.8 * len(O))
    O_train = O[:train_size]
    O_test = O[train_size:]
    actual_train = weather_df['weather_encoded'].values[:train_size]
    actual_test = weather_df['weather_encoded'].values[train_size:]

    # Initialize and train the HMM
    num_states = len(weather_mapping)
    num_observations = O.max() + 1
    hmm = HiddenMarkovModel(num_states, num_observations)

    # TODO: Train the HMM using Baum-Welch on training data
    A_trained, B_trained, pi_trained = hmm.baum_welch_log(O_train)  

    # TODO: Use the Viterbi algorithm for decoding the most likely state sequence
    train_predicted_states = hmm.viterbi_algorithm_log(O_train)  
    test_predicted_states = hmm.viterbi_algorithm_log(O_test)  

    # Decode predicted states
    train_decoded_states = [reverse_weather_mapping[state] for state in train_predicted_states]  
    test_decoded_states = [reverse_weather_mapping[state] for state in test_predicted_states]  

    # # TODO: Create and display comparison DataFrames for train and test sets
    train_comparison = pd.DataFrame({
        'Actual': [reverse_weather_mapping[state] for state in actual_train],
        'Predicted': train_decoded_states
    })
    test_comparison = pd.DataFrame({
        'Actual': [reverse_weather_mapping[state] for state in actual_test],
        'Predicted': test_decoded_states
    })
    print("\nTraining Set Comparison:\n", train_comparison.head())
    print("\nTesting Set Comparison:\n", test_comparison.head())

    # TODO: Evaluate accuracy for training and testing sets
    train_accuracy = accuracy_score(actual_train, train_predicted_states)  
    test_accuracy = accuracy_score(actual_test, test_predicted_states)  

    print("\nHMM Performance:")
    print(f"Training Set Accuracy: {train_accuracy:.2%}")
    print(f"Testing Set Accuracy: {test_accuracy:.2%}")