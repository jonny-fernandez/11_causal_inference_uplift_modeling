import pandas as pd
from src.data_contract import load_data

def main():
    df = load_data()
    print("Rows:", len(df))
    print("Treatment rate:", f"{df.training_assigned.mean():.1%}")
    print("Outcome rate:", f"{df.retained_or_promoted_180d.mean():.1%}")
    print("Raw outcome by treatment:")
    print(df.groupby("training_assigned").retained_or_promoted_180d.agg(["count", "mean"]).to_string())
    policy = pd.read_csv("data/reference/variable_policy.csv")
    print("\nVariable policy:")
    print(policy.policy.value_counts().to_string())

if __name__ == "__main__":
    main()
