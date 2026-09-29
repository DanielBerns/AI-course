import pandas as pd
import numpy as np

# Ingest / synthetic creation for standalone demo
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "created": pd.date_range("2026-01-01", periods=100, freq="h"),
    "amount": rng.uniform(10, 200, 100),
    "quantity": rng.integers(1, 5, 100),
    "region": rng.choice(["North", "South", "East", "West"], 100)
})
df = df.dropna(subset=["amount"])           # basic cleaning
df["revenue"] = df["amount"] * df["quantity"]
summary = df.groupby("region")["revenue"].agg(["count", "mean", "sum"])
print(summary)
