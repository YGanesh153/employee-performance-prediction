import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def run():
    print("🔍 Hyperparameter Tuning Started (FAST MODE)")

    df = pd.read_csv("data/processed/employee_data_featured.csv")

    le = LabelEncoder()
    df["role"] = le.fit_transform(df["role"])

    X = df.drop(columns=["performance_percentage"])
    y = df["performance_percentage"]

    y_class = np.zeros(len(y), dtype=int)
    y_class[y >= 60] = 1
    y_class[y >= 80] = 2

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # SMALL GRID → FAST
    param_grid = {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3]
    }

    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X, y_class)

    print("✅ Best Hyperparameters:")
    print(grid.best_params_)
    print("Best CV Score:", grid.best_score_)
