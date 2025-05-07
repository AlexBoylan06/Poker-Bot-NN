import shutil
import os

# Paths (edit these if your structure is different)
source_path = "../PokerBot/data/poker_data.csv"  # relative path from Poker-Bot-NN
destination_dir = "data"
destination_path = os.path.join(destination_dir, "poker_data.csv")

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Move the file
try:
    shutil.copy(source_path, destination_path)
    print(f"✅ File copied to {destination_path}")
except FileNotFoundError:
    print("❌ Source file not found. Make sure you've generated the data in PokerBot.")
except Exception as e:
    print(f"❌ Error: {e}")
