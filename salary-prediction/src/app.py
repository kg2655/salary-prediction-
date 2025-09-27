from flask import Flask, request, render_template
import joblib
import pandas as pd
from data_prep import default_feature_engineering  # Must match training

app = Flask(__name__)

# Load trained model
model = joblib.load("models/salary_model.joblib")

# Load dataset to populate dropdowns
df = pd.read_csv("data/salaries.csv")
df = default_feature_engineering(df)

job_titles = df['job_title'].dropna().unique().tolist()
locations = df['location'].dropna().unique().tolist()
education_levels = df['education_level'].dropna().unique().tolist()

@app.route("/")
def home():
    return render_template(
        "index.html",
        job_titles=job_titles,
        locations=locations,
        education_levels=education_levels
    )

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input values
        years_experience = float(request.form["years_experience"])
        job_title = request.form["job_title"]
        education_level = request.form["education_level"]
        location = request.form["location"]
        skills = request.form.get("skills", "")
        skills_count = 0 if skills.strip() == "" else len([s for s in skills.split(",") if s.strip()])

        # Validate / cap numeric inputs
        years_experience = max(0, min(years_experience, 50))  # realistic range
        skills_count = max(0, min(skills_count, 10))          # cap to max skills in training

        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "years_experience": years_experience,
            "job_title": job_title,
            "education_level": education_level,
            "location": location,
            "skills": skills,
            "skills_count": skills_count
        }])

        # Apply same feature engineering as training
        input_df = default_feature_engineering(input_df)

        # Predict using model
        prediction = model.predict(input_df)[0]

        # Remove np.exp if target was NOT log-transformed
        # If your model was trained on log(salary), uncomment the next line:
        # prediction = np.exp(prediction)

        # Round to nearest 1000
        prediction = round(prediction, -3)

        # Display as currency
        prediction_text = f"Estimated Salary: ₹{prediction:,.0f}"

        return render_template(
            "index.html",
            prediction_text=prediction_text,
            job_titles=job_titles,
            locations=locations,
            education_levels=education_levels
        )

if __name__ == "__main__":
    app.run(debug=True)
