# Adobe Sentiment Classifier

## Authors:

## Overview
Brief summary of the project — what it does, why it exists, and who it’s for.

---
## Tech Stack

---

## Features
- [ ] Core functionality (e.g., sentiment classification)
- [ ] Adobe-specific preprocessing
- [ ] Custom label support (optional)
- [ ] Model evaluation + aggregation
- [ ] Integration with engagement metrics

---
## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/adobe-sentiment.git
cd adobe-sentiment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
no need to clean data again, do not touch!! 
You only have to split the data with your respective cleaned csv file


```bash
python src/split_and_clean.py
```
---

## Project Structure
```bash
├── data/              # Input datasets (CSV, JSON, etc.)
├── models/            # Trained models and checkpoints
├── notebooks/         # Analysis and evaluation notebooks
├── src/               # Source code
│   ├── train.py       # Training loop
│   ├── preprocess.py  # Data cleaning and normalization
│   └── utils.py       # Metrics, helpers
├── requirements.txt   # Python dependencies
└── README.md
```
---
## Challenges
---
## Future Enhancements
