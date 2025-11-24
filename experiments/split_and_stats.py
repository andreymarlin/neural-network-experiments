# experiments/split_and_stats.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("tapaco_paraphrases_dataset.csv")

# Read as TAB-separated file
df = pd.read_csv(
    DATA_PATH,
    sep="\t",          # 👈 KEY CHANGE
    engine="python",
    on_bad_lines="skip"
)

print("Original columns:", df.columns)
print("Total rows:", len(df))
print(df.head())

# Rename columns to something convenient
# Automatically grab the first two columns
text_col, para_col = df.columns[:2]

df = df[[text_col, para_col]].rename(
    columns={text_col: "sentence1", para_col: "sentence2"}
)

print("\nRenamed columns:", df.columns)
print(df.head())

# ---- train/val/test split ----
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

print("\nSizes -> Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nSaved: train.csv, val.csv, test.csv")
