import os
import joblib
import pandas as pd

def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "..", "results", "trained_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model.py first.")
    return joblib.load(model_path)

def generate_workout_recommendation(fitness_goal, calories_pred):
    """Maps user objective and predicted energy expenditure to a workout routine."""
    goal = str(fitness_goal).strip().lower()
    
    if 'loss' in goal:
        split = "Cardio & Full Body HIIT Circuit"
        intensity = "High Heart Rate Zone (75-85% max BPM)"
        frequency = "4-5 days/week"
    elif 'muscle' in goal or 'hypertrophy' in goal:
        split = "Upper / Lower or Push-Pull-Legs (PPL) Resistance Split"
        intensity = "Moderate Dynamic (65-75% max BPM, progressive overload)"
        frequency = "4 days/week"
    elif 'endurance' in goal:
        split = "Aerobic Base Conditioning & Interval Running"
        intensity = "Aerobic Zone (70-80% max BPM)"
        frequency = "4-6 days/week"
    elif 'strength' in goal:
        split = "Compound Lifting (5x5 Squat, Bench, Deadlift, Overhead Press)"
        intensity = "High Neuromuscular Load with 3-minute rest periods"
        frequency = "3-4 days/week"
    else:
        split = "Functional Mobility, Calisthenics & Active Recovery"
        intensity = "Low-Impact Zone (55-65% max BPM)"
        frequency = "3 days/week"
        
    return {
        'Predicted_Calories_kcal': round(calories_pred, 1),
        'Recommended_Routine': split,
        'Target_Intensity_Zone': intensity,
        'Recommended_Frequency': frequency
    }

def predict_single_user(user_dict):
    pipe = load_pipeline()
    df_user = pd.DataFrame([user_dict])
    
    # Feature engineering for input
    if 'BMI' not in df_user.columns and 'Weight_kg' in df_user.columns and 'Height_m' in df_user.columns:
        df_user['BMI'] = (df_user['Weight_kg'] / (df_user['Height_m'] ** 2)).round(2)
        
    pred_calories = pipe.predict(df_user)[0]
    goal = user_dict.get('Fitness_Goal', user_dict.get('Workout_Type', 'General Fitness'))
    return generate_workout_recommendation(goal, pred_calories)

if __name__ == "__main__":
    # Sample Test User
    sample_user = {
        'Age': 29,
        'Gender': 'Male',
        'Weight_kg': 82.5,
        'Height_m': 1.80,
        'Session_Duration_min': 87,
        'Workout_Type': 'HIIT',
        'Fitness_Goal': 'Weight Loss',
        'Avg_BPM': 148,
        'Max_BPM': 176,
        'Workout_Frequency': 4,
        'Experience_Level': 'Intermediate'
    }
    
    print("Testing single-user prediction...")
    try:
        output = predict_single_user(sample_user)
        print("\n--- Recommendation Output ---")
        for k, v in output.items():
            print(f"{k}: {v}")
    except FileNotFoundError as e:
        print(e)
