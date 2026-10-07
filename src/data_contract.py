import pandas as pd
REQUIRED={"person_id","training_assigned","retained_or_promoted_180d","baseline_performance","baseline_engagement"}
def load_data(path="data/raw/training_intervention_synthetic.csv"):
    df=pd.read_csv(path); validate_data(df); return df
def validate_data(df):
    m=REQUIRED-set(df.columns)
    if m: raise ValueError(f"Missing required columns: {sorted(m)}")
    if df.person_id.duplicated().any(): raise ValueError("person_id must be unique")
    for c in ["training_assigned","retained_or_promoted_180d"]:
        if not set(df[c].unique()).issubset({0,1}): raise ValueError(f"{c} must be binary")
    if df.training_assigned.mean() in (0,1): raise ValueError("Both treatment groups are required")
    return True
