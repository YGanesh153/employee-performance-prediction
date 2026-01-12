from src import (
    data_cleaning,
    eda_comparison,
    feature_engineering,
    baseline_model,
    model_training
)

import pandas as pd
import joblib

# ================= OPTIONAL ANALYSIS FLAGS =================
# ⚠️ THESE ARE PYTHON VARIABLES (DO NOT TYPE IN TERMINAL)
RUN_HYPERPARAMETER_TUNING = True
RUN_STATISTICAL_VALIDATION = True
# ===========================================================

def main():

    print("=" * 60)
    print("EMPLOYEE PERFORMANCE PREDICTION PIPELINE")
    print("=" * 60)

    print("\n[STEP 1] DATA CLEANING")
    data_cleaning.run()

    print("\n[STEP 2] EXPLORATORY DATA ANALYSIS")
    eda_comparison.run()

    print("\n[STEP 3] FEATURE ENGINEERING")
    feature_engineering.run()

    # ======================================================
    # OPTIONAL OFFLINE ANALYSIS (SHOWS OUTPUT)
    # ======================================================
    if RUN_HYPERPARAMETER_TUNING:
        print("\n[OPTIONAL] HYPERPARAMETER TUNING OUTPUT")
        from src.hyperparameter_tuning import run as tune
        tune()

    if RUN_STATISTICAL_VALIDATION:
        print("\n[OPTIONAL] STATISTICAL VALIDATION OUTPUT")
        from src.statistical_validation import run as stats
        stats()
    # ======================================================

    print("\n[STEP 4] BASELINE MODEL TRAINING")
    baseline_model.run()

    print("\n[STEP 5] FINAL MODEL TRAINING & SAVING")
    model_training.run()

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTED SUCCESSFULLY ✅")
    print("MODEL READY FOR DEPLOYMENT")
    print("=" * 60)

    # -------- LOAD FINAL MODEL --------
    model = joblib.load("model/performance_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    role_encoder = joblib.load("model/role_encoder.pkl")

    while True:
        print("\n[ENTER EMPLOYEE DETAILS]")

        attendance = float(input("Attendance Consistency (%): "))
        task_completion = float(input("Task Completion (%): "))
        task_delay = float(input("Task Delay Rate (%): "))
        productive_hours = float(input("Productive Hours: "))
        experience = float(input("Experience (years): "))
        project_complexity = float(input("Project Complexity (1–5): "))
        feedback = float(input("Feedback Score (1–5): "))

        print("\nAvailable Roles:")
        for r in role_encoder.classes_:
            print("-", r)

        role = input("Enter Role exactly as shown: ")

        if role not in role_encoder.classes_:
            print("❌ Invalid role. Try again.")
            continue

        role_encoded = role_encoder.transform([role])[0]

        X_input = pd.DataFrame([{
            "attendance_consistency": attendance,
            "task_completion": task_completion,
            "task_delay_rate": task_delay,
            "productive_hours": productive_hours,
            "experience": experience,
            "project_complexity": project_complexity,
            "feedback_score": feedback,
            "efficiency_score": task_completion - task_delay,
            "exp_productivity": productive_hours * (1 + experience / 10),
            "role": role_encoded
        }])

        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]

        # -------- BUSINESS RULE FIX --------
        if task_completion < 30 or feedback < 3:
            result = "Low Performer → Improvement Plan"
        else:
            if pred == 0:
                result = "Low Performer → Improvement Plan"
            elif pred == 1:
                result = "Average Performer → ₹2,000 Bonus"
            else:
                result = "High Performer → ₹10,000 Bonus + Promotion"

        print("\nPrediction Result:", result)

        again = input("\nPredict another employee? (y/n): ").lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()
