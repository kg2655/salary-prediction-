# Salary Prediction

This project uses machine learning to predict employee salaries based on various features such as experience, education, job title, location, and company.

## Features

- Data preprocessing and feature engineering
- Model training using scikit-learn (RandomForestRegressor)
- Evaluation metrics: MAE, RMSE, R²
- Easily customizable for different datasets

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kg2655/salary-prediction.git
    cd salary-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Train the model with your dataset:
```bash
python src/train.py --data data/salaries.csv --out models/salary_model.joblib
```

## Project Structure

```
salary-prediction/
├─ data/
│ └─ salaries.csv # (example) your dataset (see format below)
├─ notebooks/
│ └─ EDA_and_training.ipynb
├─ src/
│ ├─ data_prep.py
│ ├─ train.py
│ ├─ model.py
│ └─ app.py # Flask REST API for predictions
├─ examples/
│ └─ sample_request.json
├─ requirements.txt
└─ README.md
```

## License

This project is licensed under the MIT License.
