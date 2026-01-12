from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/performance_model.pkl")
scaler = joblib.load("model/scaler.pkl")
role_encoder = joblib.load("model/role_encoder.pkl")


def get_decision(cls):
    if cls == 0:
        return "Low Performer → Improvement Plan"
    elif cls == 1:
        return "Average Performer → ₹2,000 Bonus"
    else:
        return "High Performer → ₹10,000 Bonus + Promotion"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form values
        attendance = float(request.form["attendance"])
        task_completion = float(request.form["task_completion"])
        task_delay = float(request.form["task_delay"])
        productive_hours = float(request.form["productive_hours"])
        experience = float(request.form["experience"])
        project_complexity = float(request.form["project_complexity"])
        feedback = float(request.form["feedback"])
        role = request.form["role"]

        # Encode role
        role_encoded = role_encoder.transform([role])[0]

        # Feature engineering (same as training)
        efficiency_score = task_completion - task_delay
        exp_productivity = productive_hours * (1 + experience / 10)

        X_input = pd.DataFrame([{
            "attendance_consistency": attendance,
            "task_completion": task_completion,
            "task_delay_rate": task_delay,
            "productive_hours": productive_hours,
            "experience": experience,
            "project_complexity": project_complexity,
            "feedback_score": feedback,
            "efficiency_score": efficiency_score,
            "exp_productivity": exp_productivity,
            "role": role_encoded
        }])

        # Scale input
        X_scaled = scaler.transform(X_input)

        # Predict
        pred_class = model.predict(X_scaled)[0]

        # Get business decision
        result = get_decision(pred_class)

        return render_template("result.html", result=result)

    # GET request
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
