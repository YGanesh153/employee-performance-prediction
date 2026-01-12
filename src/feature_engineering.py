import pandas as pd
import os

def run():
    print("Feature Engineering Started...")

    df = pd.read_csv("data/cleaned/employee_data_cleaned.csv")

    # Engineered features (SAFE)
    df["efficiency_score"] = df["task_completion"] - df["task_delay_rate"]
    df["exp_productivity"] = df["productive_hours"] * (1 + df["experience"] / 10)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/employee_data_featured.csv", index=False)

    print("Feature Engineering Completed.")
