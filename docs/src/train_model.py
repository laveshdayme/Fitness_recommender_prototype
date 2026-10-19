import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate():
    # 1. Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gym_members_workout.csv")
    df = pd.read_csv(data_path)
    
    # 2. Feature Engineering
    if 'BMI' not in df.columns and 'Weight_kg' in df.columns and 'Height_m' in df.columns:
        df['BMI'] = (df['Weight_kg'] / (df['Height_m'] ** 2)).round(2)
        
    # Standardize column naming if needed
    target_col = 'Calories_Burned'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Preprocessing Setup
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])
    
    # 4. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 5. Model Definitions
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring='r2').mean()
        
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_r2})
        print(f"{name:20s} | Test R2: {r2:.4f} | RMSE: {rmse:.2f} kcal | CV R2: {cv_r2:.4f}")
        
    # 6. Save Results & Best Pipeline (Linear Regression)
    res_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(res_dir, "model_performance.csv"), index=False)
    
    best_pipe = trained_pipelines['Linear Regression']
    joblib.dump(best_pipe, os.path.join(res_dir, "trained_pipeline.joblib"))
    print("\nSaved best model pipeline to results/trained_pipeline.joblib")

if __name__ == "__main__":
    train_and_evaluate()
