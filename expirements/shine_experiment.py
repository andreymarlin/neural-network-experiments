import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
print("hi i am running")
df = pd.read_csv(
    "/Users/shinekhantaung/Documents/dcs_project/neural-network-experiments/data/tapaco_paraphrases_dataset.csv",
    on_bad_lines='skip'
)

# Split: 80% train, 10% val, 10% test
train, test = train_test_split(df, test_size=0.20, random_state=42)
train, val = train_test_split(train, test_size=0.10, random_state=42)

# Save splits

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Dataset split completed successfully!")
