import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('poker_data.csv')

# Check column names
print("Columns:", df.columns.tolist())

# Define value and suit dictionaries
values = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
          '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
suits = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}

# Function to one-hot encode a card
def one_hot_card(card):
    if not isinstance(card, str) or ' of ' not in card:
        print(f"Skipping invalid card: {card}")
        return [0] * 52
    value_str, suit_str = card.split(' of ')
    value_idx = values.get(value_str)
    suit_idx = suits.get(suit_str)
    if value_idx is None or suit_idx is None:
        print(f"Unrecognized card: {card}")
        return [0] * 52
    index = value_idx * 4 + suit_idx
    one_hot = [0] * 52
    one_hot[index] = 1
    return one_hot

# Split board into five separate cards
df[['board1', 'board2', 'board3', 'board4', 'board5']] = df['board'].str.split(r' \| ', expand=True)

# One-hot encode all 7 cards
one_hot_df = pd.DataFrame()
for col in ['card1', 'card2', 'board1', 'board2', 'board3', 'board4', 'board5']:
    one_hot_matrix = df[col].apply(one_hot_card).apply(pd.Series)
    one_hot_matrix.columns = [f"{col}_{i}" for i in range(52)]
    one_hot_df = pd.concat([one_hot_df, one_hot_matrix], axis=1)

# Add score as label
if 'score' in df.columns:
    one_hot_df['label'] = df['score']

# Save
one_hot_df.to_csv('poker_data_onehot.csv', index=False)
print("✅ One-hot encoded file saved as 'poker_data_onehot.csv'")
