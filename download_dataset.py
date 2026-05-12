import pandas as pd

print("Caricando il dataset...")
df = pd.read_csv("data/amazon_review_polarity_csv/train.csv", header=None, names=["label", "title", "text"])

print(f"Record totali: {len(df)}")
df = df[df['text'].str.len() > 50].dropna(subset=['text'])
df = df.sample(1500, random_state=42)

train = df.sample(1200, random_state=42)
frozen = df.drop(train.index)

train.to_csv('data/train.csv', index=False)
frozen.to_csv('data/frozen_eval.csv', index=False)

print(f"Fatto! Train: {len(train)} righe, Frozen: {len(frozen)} righe")
print(train['text'].iloc[0])