import os
import pytest
import pandas as pd
from src.predict import predict_single_user, generate_workout_recommendation

def test_prediction_output_range_and_types():
    """
    Test that the prediction returns the expected dictionary keys,
    data types, and that predicted calories fall within physiologically plausible bounds.
    """
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
    
    result = predict_single_user(sample_user)
    
    # 1. Verify schema keys
    assert isinstance(result, dict)
    assert 'Predicted_Calories_kcal' in result
    assert 'Recommended_Routine' in result
    assert 'Target_Intensity_Zone' in result
    assert 'Recommended_Frequency' in result
    
    # 2. Check physiological calorie boundaries (100 to 2000 kcal)
    calories = result['Predicted_Calories_kcal']
    assert isinstance(calories, (int, float))
    assert 100 <= calories <= 2000, f"Unrealistic calorie estimate: {calories}"

def test_recommendation_logic():
    """
    Test that the workout rule-mapping returns correct routines based on fitness goals.
    """
    # Weight loss scenario
    wl_rec = generate_workout_recommendation('Weight Loss', 950.0)
    assert "HIIT" in wl_rec['Recommended_Routine'] or "Cardio" in wl_rec['Recommended_Routine']
    
    # Muscle gain scenario
    mg_rec = generate_workout_recommendation('Muscle Gain', 650.0)
    assert "Resistance" in mg_rec['Recommended_Routine'] or "PPL" in mg_rec['Recommended_Routine']

def test_edge_case_duration():
    """
    Test edge-case inputs (short vs long workouts) to verify stability.
    """
    short_session_user = {
        'Age': 22,
        'Gender': 'Female',
        'Weight_kg': 55.0,
        'Height_m': 1.62,
        'Session_Duration_min': 20,
        'Workout_Type': 'Yoga',
        'Fitness_Goal': 'Flexibility',
        'Avg_BPM': 110,
        'Max_BPM': 130,
        'Workout_Frequency': 2,
        'Experience_Level': 'Beginner'
    }
    
    result = predict_single_user(short_session_user)
    assert result['Predicted_Calories_kcal'] > 50
