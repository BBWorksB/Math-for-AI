import random
import matplotlib.pyplot as plt

# Define the deck of cards
suits = ['hearts', 'diamonds', 'clubs', 'spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
deck = [(rank, suit) for suit in suits for rank in ranks]

# Theoretical probabilities
total_cards = 52
hearts_count = 13  # 13 hearts in a deck
face_card_count = 12  # 3 face cards (Jack, Queen, King) in each of the 4 suits

prob_heart_theory = hearts_count / total_cards
prob_face_theory = face_card_count / total_cards

# Function to calculate probabilities in a sample
def calculate_sample_probabilities(sample):
    hearts_in_sample = sum(1 for card in sample if card[1] == 'hearts')
    face_cards_in_sample = sum(1 for card in sample if card[0] in ['Jack', 'Queen', 'King'])
    return hearts_in_sample / len(sample), face_cards_in_sample / len(sample)

# Simulate drawing cards without replacement and compute probabilities
sample_size_range = list(range(1, 53))
prob_heart_simulation = []
prob_face_simulation = []

# Run simulations for sample sizes from 1 to 52
for sample_size in sample_size_range:
    sample = random.sample(deck, sample_size)
    prob_heart, prob_face = calculate_sample_probabilities(sample)
    prob_heart_simulation.append(prob_heart)
    prob_face_simulation.append(prob_face)

# Plot theoretical and simulated probabilities for hearts and face cards
plt.figure(figsize=(10, 6))

# Plot hearts probabilities
plt.plot(sample_size_range, prob_heart_simulation, label='Simulated Heart Probability', marker='o')
plt.axhline(y=prob_heart_theory, color='r', linestyle='-', label='Theoretical Heart Probability')

# Plot face card probabilities
plt.plot(sample_size_range, prob_face_simulation, label='Simulated Face Card Probability', marker='x')
plt.axhline(y=prob_face_theory, color='g', linestyle='-', label='Theoretical Face Card Probability')

plt.xlabel('Sample Size')
plt.ylabel('Probability')
plt.title('Simulated vs Theoretical Probabilities (Hearts and Face Cards)')
plt.legend()
plt.grid(True)
plt.show()
