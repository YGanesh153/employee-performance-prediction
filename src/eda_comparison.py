import pandas as pd

def run():
    print("EDA Started...")

    df = pd.read_csv("data/processed/cleaned_data.csv")

    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescribe:\n", df.describe())

    print("EDA Completed.")
