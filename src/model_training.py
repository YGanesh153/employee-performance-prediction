import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def run():
    print("Final Model Training Started...")

    df = pd.read_csv("data/processed/employee_data_featured.csv")

    role_encoder = LabelEncoder()
    df["role"] = role_encoder.fit_transform(df["role"])

    features = [
        "attendance_consistency",
        "task_completion",
        "task_delay_rate",
        "productive_hours",
        "experience",
        "project_complexity",
        "feedback_score",
        "efficiency_score",
        "exp_productivity",
        "role"
    ]

    X = df[features]
    y = df["performance_percentage"]

    y_class = np.zeros(len(y), dtype=int)
    y_class[y >= 60] = 1
    y_class[y >= 80] = 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nFinal Model Evaluation:\n")
    print(classification_report(y_test, preds))

    # Feature importance
    importances = model.feature_importances_
    plt.barh(features, importances)
    plt.title("Feature Importance - Gradient Boosting")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/performance_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(role_encoder, "model/role_encoder.pkl")

    print("Final Model Training Completed.")
