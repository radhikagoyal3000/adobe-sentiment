import re

def preprocess_text(text):
    # Included other applciations because we are unable to filter on Instagram for now
    text = re.sub(r"\b(photoshop|firefly|lightroom|premiere|illustrator)\b", "[PRODUCT]", text, flags=re.IGNORECASE)
    text = re.sub(r"v\d+(\.\d+)*", "[VERSION]", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s\[\]]", "", text)
    return text.strip()
