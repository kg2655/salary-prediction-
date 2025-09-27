"""Helper to load the saved pipeline and make predictions."""
import joblib
import pandas as pd




class SalaryPredictor:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)


def predict_single(self, payload: dict):
    df = pd.DataFrame([payload])
    preds = self.pipeline.predict(df)
    return float(preds[0])


def predict_batch(self, dataframe: pd.DataFrame):
    preds = self.pipeline.predict(dataframe)
    return preds




if __name__ == '__main__':
    p = SalaryPredictor('models/salary_model.joblib')
    example = {
    'job_title': 'data scientist',
    'years_experience': 3,
    'education_level': 'Masters',
    'location': 'Bengaluru',
    'skills': 'python,ml,sql'
    }
    print('Predicted salary:', p.predict_single(example))