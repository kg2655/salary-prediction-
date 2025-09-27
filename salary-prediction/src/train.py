"""Train a regression model for salary prediction and export it."""
import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


from data_prep import load_data, build_preprocessor, train_test_split_features




def create_and_train(data_path, model_out='models/salary_model.joblib', test_size=0.2):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)


    df = load_data(data_path)


# basic feature list detection
    numeric_features = []
    possible_numeric = ['years_experience', 'skills_count', 'experience']
    for c in possible_numeric:
        if c in df.columns:
            numeric_features.append(c)
# if years_experience not present, try to infer
    if 'years_experience' not in numeric_features and 'years_experience' in df.columns:
        numeric_features.append('years_experience')


    categorical_features = []
    for c in ['job_title', 'education_level', 'location', 'company']:
        if c in df.columns:
            categorical_features.append(c)


    if len(numeric_features) == 0:
# if no obvious numeric features, try to coerce numeric columns
        for c in df.select_dtypes(include=[np.number]).columns:
            if c != 'salary':
                numeric_features.append(c)


    print('Numeric features:', numeric_features)
    print('Categorical features:', categorical_features)


    X_train, X_test, y_train, y_test = train_test_split_features(df, target_col='salary', test_size=test_size)


    preprocessor = build_preprocessor(categorical_features, numeric_features)


    model = RandomForestRegressor(n_estimators=100, random_state=42)


    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])


    print('Fitting model...')
    pipeline.fit(X_train, y_train)


    print('Predicting on test set...')
    preds = pipeline.predict(X_test)


    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    r2 = r2_score(y_test, preds)


    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")


    joblib.dump(pipeline, model_out)
    print(f'Saved trained pipeline to {model_out}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    parser.add_argument('--out', default='models/salary_model.joblib', help='Output path for trained model')
    args = parser.parse_args()
    create_and_train(args.data, args.out)