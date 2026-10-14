"""
run_experiments.py
-----------------
Running: -
.\.venv\Scripts\python.exe run_experiments
"""


from src.data_loader import load_dataset
from src.preprocessing import prepare_dataset, build_preprocessor
from src.feature_groups import prepare_experiment

# Load the career dataset.
df = load_dataset("career_data")

# Separate features and target.
X, y = prepare_dataset(
    df,
    dataset_key="career_data",
)

# Select features and split the dataset.
result = prepare_experiment(
    X=X,
    y=y,
    dataset_key="career_data",
    experiment_name="CD_academic_plus_skills",
    test_size=0.20,
    random_state=42,
)

# Fit preprocessing ONLY on the training data.
preprocessor = build_preprocessor(result["X_train"])

X_train = preprocessor.fit_transform(result["X_train"])
X_test = preprocessor.transform(result["X_test"])

y_train = result["y_train"]
y_test = result["y_test"]

print("Experiment:", result["experiment_name"])

"""
print("Features:", result["feature_columns"]) 
"""
"""
Fix the RIASEC sensitivity experiment: -
Only run this sensitivity experiment after checking how the Career labels were generated."""

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
