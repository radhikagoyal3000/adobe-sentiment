import pandas as pd
import re
import emoji
from langdetect import detect, DetectorFactory

# Ensure consistent and deterministic language detection
DetectorFactory.seed = 0

# List of Adobe product keywords to detect or mask
ADOBE_PRODUCTS = [
    "photoshop", "lightroom", "premiere", "illustrator", "firefly",
    "after effects", "indesign", "audition", "xd", "acrobat", "animate", "bridge"
]

# === Clean the comment text ===
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = emoji.demojize(text, language='en')                       # 🔥 → :fire:
    text = re.sub(r"v\d+(\.\d+)*", "[version]", text)                # Replace version numbers
    text = re.sub(r"http\S+", "", text)                              # Remove URLs
    text = re.sub(r"[^\w\s\[\]:]", "", text)                         # Remove punctuation except brackets and colons
    text = re.sub(r"\s+", " ", text)                                 # Normalize whitespace
    return text.strip()

# === Mask product names with [product] ===
def mask_products(text):
    for product in ADOBE_PRODUCTS:
        text = re.sub(rf"\b{re.escape(product)}\b", "[product]", text, flags=re.IGNORECASE)
    return text

# === Extract actual product mentions (unmasked) ===
def extract_products(text):
    if pd.isna(text):
        return []
    text = text.lower()
    found = []
    for product in ADOBE_PRODUCTS:
        if re.search(rf"\b{re.escape(product)}\b", text):
            found.append(product)
    return list(set(found))

# === Safe language detection (skip short or junk text) ===
def detect_language(text):
    try:
        if isinstance(text, str):
            word_count = len(text.split())
            if len(text.strip()) < 5 or word_count < 2:
                return "unknown"
            return detect(text)
        return "unknown"
    except:
        return "error"

# === Main processing function ===
def clean_youtube_comments(input_csv="data/all_youtube.csv", output_csv="data/comments_cleaned_full.csv"):
    df = pd.read_csv(input_csv)

    df["Comment Text"] = df["Comment Text"].astype(str)
    df["cleaned_comment"] = df["Comment Text"].apply(clean_text)
    df["cleaned_comment_for_sentiment"] = df["cleaned_comment"].apply(mask_products)
    df["mentioned_products"] = df["Comment Text"].apply(extract_products)
    df["language"] = df["cleaned_comment"].apply(detect_language)
    df["is_english"] = df["language"] == "en"

    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned YouTube comments saved to: {output_csv}")
    print(f"📊 Columns include: cleaned_comment, cleaned_comment_for_sentiment, mentioned_products, language, is_english")

# === Run it ===
if __name__ == "__main__":
    clean_youtube_comments()
