import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgb_best_model.pkl')

# Define the feature columns in the correct order that the model was trained on
model_columns = [
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M',
    'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL',
    'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL',
    'Age_Newest_TL', 'time_since_recent_payment',
    'max_recent_level_of_deliq', 'num_deliq_6_12mts',
    'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts',
    'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss',
    'recent_level_of_deliq', 'CC_enq_L12m', 'PL_enq_L12m',
    'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
    'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag',
    'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever', 'HL_Flag',
    'GL_Flag', 'EDUCATION', 'MARITALSTATUS_Married', 'MARITALSTATUS_Single',
    'GENDER_F', 'GENDER_M', 'last_prod_enq2_AL', 'last_prod_enq2_CC',
    'last_prod_enq2_ConsumerLoan', 'last_prod_enq2_HL',
    'last_prod_enq2_PL', 'last_prod_enq2_others', 'first_prod_enq2_AL',
    'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan',
    'first_prod_enq2_HL', 'first_prod_enq2_PL', 'first_prod_enq2_others'
]

# Mapping for the prediction labels
label_map = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # Convert incoming JSON data to a pandas DataFrame
        input_df = pd.DataFrame([data])

        # --- Data Preprocessing ---

        # 1. Map EDUCATION feature as done in the notebook
        education_map = {
            'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
            'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
        }
        input_df['EDUCATION'] = input_df['EDUCATION'].map(education_map)

        # 2. Convert all numerical columns to the correct data type
        for col in input_df.columns:
            if col != 'MARITALSTATUS' and col != 'GENDER' and col != 'last_prod_enq2' and col != 'first_prod_enq2':
                 input_df[col] = pd.to_numeric(input_df[col], errors='coerce')


        # 3. One-Hot Encode categorical variables
        # Use get_dummies and align with the model's columns to ensure consistency
        input_encoded = pd.get_dummies(input_df)
        
        # Align the columns of the input data with the model's training columns
        # This adds missing columns with a value of 0 and removes extra columns
        model_input, _ = input_encoded.align(pd.DataFrame(columns=model_columns), join='right', axis=1, fill_value=0)
        
        # Ensure the final column order matches the model's expectation
        model_input = model_input[model_columns]

        # --- Prediction ---
        prediction_encoded = model.predict(model_input)
        
        # Map the numeric prediction back to the original label
        prediction = label_map[prediction_encoded[0]]

        # Return the result as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
