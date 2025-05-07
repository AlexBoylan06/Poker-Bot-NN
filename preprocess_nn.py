import pandas as pd
from treys import Card
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = 'poker_data.csv'
OUTPUT_DIR = 'processed_data'
CARD_COLUMNS = ['hole1', 'hole2', 'flop1', 'flop2', 'flop3', 'turn', 'river']
TARGET_COLUMN = 'win_prob'


# -----------------------------
# UTILS
# -----------------------------
def encode_card(card_str):
    rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                '8': '8', '9': '9', '10': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
    suit_map = {'Spades': 's', 'Hearts': 'h', 'Diamonds': 'd', 'Clubs': 'c'}
    try:
        rank, _, suit = card_str.partition(' of ')
        treys_str = rank_map[rank] + suit_map[suit]
        return Card.new(treys_str)
    except:
        return 0  # default/fallback value


# -----------------------------
# LOAD & CLEAN
# -----------------------------
def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df = df.dropna()
    return df


# -----------------------------
# ENCODE + NORMALIZE
# -----------------------------
def preprocess(df):
    print("Encoding cards...")
    for col in CARD_COLUMNS:
        df[col] = df[col].apply(encode_card)

    print("Normalizing target column...")
    scaler = MinMaxScaler()
    df[[TARGET_COLUMN]] = scaler.fit_transform(df[[TARGET_COLUMN]])

    print("Splitting into X and y...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y, scaler


# -----------------------------
# SAVE
# -----------------------------
def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(f'{OUTPUT_DIR}/X_train.csv', index=False)
    X_test.to_csv(f'{OUTPUT_DIR}/X_test.csv', index=False)
    y_train.to_csv(f'{OUTPUT_DIR}/y_train.csv', index=False)
    y_test.to_csv(f'{OUTPUT_DIR}/y_test.csv', index=False)
    with open(f'{OUTPUT_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Processed data saved to '{OUTPUT_DIR}'.")


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data(DATA_PATH)
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)


if __name__ == '__main__':
    main()
