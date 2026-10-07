import pandas as pd
from src.data_contract import load_data
def test_contract():
    df=load_data(); assert len(df)>=4000; assert df.person_id.is_unique
def test_binary_treatment_outcome():
    df=load_data(); assert set(df.training_assigned.unique())=={0,1}; assert set(df.retained_or_promoted_180d.unique())=={0,1}
def test_both_groups_large_enough():
    df=load_data(); counts=df.training_assigned.value_counts(); assert counts.min()>1000
def test_post_treatment_policy():
    p=pd.read_csv("data/reference/variable_policy.csv").set_index("column"); assert p.loc["post_training_completion","policy"]=="EXCLUDED_POST_TREATMENT"
def test_ground_truth_not_feature():
    p=pd.read_csv("data/reference/variable_policy.csv").set_index("column"); assert p.loc["true_individual_treatment_effect","policy"]=="EVALUATION_ONLY"
