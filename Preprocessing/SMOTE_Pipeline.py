import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 📥 Load your dataset
df = pd.read_csv("predictive_maintenance.csv")  # Update with your actual path

# 🎯 Define target and features
target_col = 'Failure Type'
X = df.drop(columns=[target_col])
y = df[target_col]

# 🧼 Optional: One-hot encode categorical features (if any)
X = pd.get_dummies(X)

# 📊 Train-test split with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 🧬 Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 🛠️ Merge resampled features and target
resampled_df = pd.concat(
    [pd.DataFrame(X_resampled, columns=X.columns),
     pd.DataFrame(y_resampled, columns=[target_col])],
    axis=1
)

# 💾 Save the resampled dataset for AutoAI upload
resampled_df.to_csv("resampled_predictive_maintenance.csv.gz", index=False, compression="gzip")