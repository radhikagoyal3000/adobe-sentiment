import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text

# Load dataset
df = pd.read_csv("data/comments.csv")

# Clean comments
df["comment"] = df["comment"].astype(str).apply(preprocess_text)

# Optional: drop rows with missing labels
df = df.dropna(subset=["comment", "label"])

# Split into train/val/test
train_val, test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train, val = train_test_split(train_val, test_size=0.111, stratify=train_val["label"], random_state=42)

# Save files
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Data successfully cleaned and split into train/val/test.")
