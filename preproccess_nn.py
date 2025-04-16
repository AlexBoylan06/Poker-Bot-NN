import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Card rank and suit mapping
RANKS = '23456789TJQKA'
SUITS = 'CDHS'
CARD_TO_INDEX = {f'{rank}{suit}': i for i, (rank, suit) in enumerate((r, s) for s in SUITS for r in RANKS)}

def encode_cards(cards):
    """Convert cards to a 52-bit one-hot vector."""
    encoding = np.zeros(52, dtype=np.float32)
    for card in cards.split(', '):
        index = CARD_TO_INDEX[card]
        encoding[index] = 1
    return encoding

def load_and_preprocess_data(file):
    # Load data
    df = pd.read_csv(file)

    # Encode hole cards and community cards
    hole_cards = df['Hole Cards'].apply(encode_cards)
    flop = df['Flop'].apply(encode_cards)
    turn = df['Turn'].apply(encode_cards)
    river = df['River'].apply(encode_cards)

    # Combine all card encodings into a single input vector
    X = np.hstack([
        np.vstack(hole_cards),
        np.vstack(flop),
        np.vstack(turn),
        np.vstack(river)
    ])

    # Encode hand rank (target)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Hand Rank'])

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Example usage:
if __name__ == "__main__":
    file = 'data/poker_data.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file)

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    print("Sample training input:", X_train[0])
    print("Sample training label:", y_train[0])
