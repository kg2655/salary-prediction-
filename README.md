# Employee Salary Prediction

## 📝 Overview

This project focuses on building a machine learning model to predict employee salaries based on various features such as job title, years of experience, education level, location, and skills. The goal is to provide an estimated salary for a given set of employee attributes.

## 📊 Dataset

**Source:** Internal company data (simulated for this project)
**File:** `salaries.csv`
**Description:** Contains salary information for various job roles.
**Key Columns:**
*   `job_title` – The professional title of the employee.
*   `years_experience` – Number of years of professional experience.
*   `education_level` – Highest level of education attained.
*   `location` – Geographical location of the job.
*   `skills` – Comma-separated list of skills possessed.
*   `salary` – The target variable: employee's salary.

## 🛠 Tools & Libraries

*   **Python**
*   **Pandas** – For data manipulation and analysis.
*   **NumPy** – For numerical operations.
*   **Scikit-learn** – For machine learning model building (e.g., `RandomForestRegressor`), preprocessing (`OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`), and pipeline creation.
*   **Joblib** – For saving and loading the trained machine learning model.
*   **Flask** – For creating the web application to serve predictions.
*   **Bootstrap** – For styling the web interface.

## 🚀 Project Structure

```
.
├── data/
│   └── salaries.csv
├── models/
│   └── salary_model.joblib
├── static/
│   └── (CSS/JS files if any)
├── templates/
│   └── index.html
├── app.py
├── data_prep.py
├── model.py
├── train.py
├── requirements.txt
└── README.md
```

*   **`data/salaries.csv`**: The dataset used for training the model.
*   **`models/salary_model.joblib`**: The trained machine learning pipeline saved as a joblib file.
*   **`templates/index.html`**: The HTML template for the web application's user interface.
*   **`app.py`**: The Flask application that serves the salary prediction web interface.
*   **`data_prep.py`**: Contains functions for data loading, feature engineering (e.g., `skills_count`, normalizing `job_title`), and building the preprocessing pipeline.
*   **`model.py`**: A utility script to load the trained model and make predictions.
*   **`train.py`**: Script responsible for loading data, performing feature engineering, training the `RandomForestRegressor` model, evaluating its performance, and saving the trained pipeline.
*   **`requirements.txt`**: Lists all Python dependencies required for the project.
*   **`README.md`**: This file, providing an overview of the project.

## ✨ Key Features & Functionality

*   **Data Preprocessing**:
    *   Converts a comma-separated string of `skills` into a numerical `skills_count`.
    *   Normalizes `job_title` to lowercase.
    *   Handles missing `education_level` values.
    *   Uses `OneHotEncoder` for categorical features and `SimpleImputer` for numerical features.
*   **Machine Learning Model**:
    *   Employs a `RandomForestRegressor` for salary prediction.
    *   The entire preprocessing and model are encapsulated within a `Pipeline` for consistency.
*   **Web Application**:
    *   A user-friendly web interface built with Flask and Bootstrap.
    *   Allows users to input various job-related details (experience, job title, education, location, skills).
    *   Displays the predicted salary in a clear format.
    *   Dropdowns for `job_title`, `education_level`, and `location` are dynamically populated from the training data.

## 🚀 Setup and Run

1.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model (if not already trained):**
    ```bash
    python train.py --data data/salaries.csv --out models/salary_model.joblib
    ```
    *This step will create the `models/salary_model.joblib` file.*

4.  **Run the Flask web application:**
    ```bash
    python app.py
    ```

5.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address shown in your terminal).

## 📈 Model Performance (Example Metrics from `train.py`)

After training, the `train.py` script outputs performance metrics on the test set. An example output might look like:

*   **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.
*   **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors, giving higher weight to larger errors.
*   **R-squared (R2):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

Example:
```
MAE: 10000.00, RMSE: 12000.00, R2: 0.9500
```
*(Note: Actual values will depend on the dataset and model training.)*
```
