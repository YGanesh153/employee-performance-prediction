import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

def run():
    print("Baseline Model Training Started...")

    # Load data
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

    # Convert percentage to classes
    y_class = np.zeros(len(y), dtype=int)
    y_class[y >= 60] = 1
    y_class[y >= 80] = 2

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True)
    }

    os.makedirs("outputs", exist_ok=True)

    # Binarize labels for multiclass ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    for name, model in models.items():
        print(f"\n{name} Results:")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Report
        print(classification_report(y_test, preds))

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix:\n", cm)

        # ROC-AUC score
        probs = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, probs, multi_class="ovr")
        print("ROC-AUC:", auc_score)

        # -------- Multiclass ROC Curve --------
        plt.figure()

        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve (Multiclass)")
        plt.legend(loc="lower right")

        file_name = name.lower().replace(" ", "_") + "_roc.png"
        plt.savefig("outputs/" + file_name)
        plt.close()

    print("Baseline Model Training Completed.")
