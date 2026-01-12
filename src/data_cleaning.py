import pandas as pd
import os

def run():
    print("Data Cleaning Started...")

    df = pd.read_csv("data/raw/employee_data.csv")

    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv("data/cleaned/employee_data_cleaned.csv", index=False)

    print("Data Cleaning Completed.")
