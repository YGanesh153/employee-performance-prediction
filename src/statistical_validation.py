import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel

def run():
    print("📊 Statistical Validation Started (ULTRA FAST MODE)")

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

    lr = LogisticRegression(max_iter=1000)
    gb = GradientBoostingClassifier(random_state=42)

    # FAST cross-validation scores
    lr_scores = cross_val_score(lr, X, y_class, cv=3, scoring="f1_weighted")
    gb_scores = cross_val_score(gb, X, y_class, cv=3, scoring="f1_weighted")

    # Paired t-test
    _, p_val = ttest_rel(lr_scores, gb_scores)

    print("✅ Logistic Regression CV Scores:", lr_scores)
    print("✅ Gradient Boosting CV Scores:", gb_scores)
    print("✅ Paired T-Test p-value:", p_val)

    print("Statistical Validation Completed\n")
