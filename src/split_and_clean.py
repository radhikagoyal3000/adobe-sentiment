import pandas as pd
from sklearn.model_selection import train_test_split


# YOU HAVE to split the data by running 'python src/split_data.py' in your terminal
# as some use the uncased csv file and others use the cased one

# ======== EDIT HERE ========
CSV_PATH = "data/name_of_ur_file.csv"  # Path to your CSV file
TEXT_COL = "cleaned_comment_for_sentiment"      # Column containing the text
LABEL_COL = "sentiment_label"                   # Change to "engagement_label" if needed
# ===========================

# Load CSV
df = pd.read_csv(CSV_PATH)

# Drop rows missing text or label
df = df.dropna(subset=[TEXT_COL, LABEL_COL])

# Split data: 80% train, 10% val, 10% test
train_val, test = train_test_split(df, test_size=0.1, stratify=df[LABEL_COL], random_state=42)
train, val = train_test_split(train_val, test_size=0.111, stratify=train_val[LABEL_COL], random_state=42)

# Save splits
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("✅ Data split complete:")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
