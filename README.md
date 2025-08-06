Credit Risk Prediction Web Application
<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
<img src="https://img.shields.io/badge/XGBoost-006600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
<img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render"/>
</p>

This project is a web application that predicts credit risk based on a user's financial and personal information. It uses a machine learning model trained on a comprehensive dataset to classify applicants into different risk categories. The front end is a clean, multi-step form designed for a great user experience, and the back end is a Flask server that handles the prediction logic.

Live Demo: https://credit-risk-predictor-02fj.onrender.com/

## Features
Multi-Step Form: A user-friendly, multi-page form that logically groups input fields for a smoother user experience.

Real-Time Prediction: Submitting the form sends the data to a Flask back end, which returns a credit risk prediction in real-time.

Machine Learning Model: Utilizes a trained XGBoost Classifier model to make accurate predictions based on the input data.

Clean & Responsive UI: The front end is built with Tailwind CSS for a modern and responsive design that works on all devices.

Deployed on Render: The application is deployed and publicly accessible via Render.

## Tech Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, XGBoost, Pandas, NumPy

Frontend: HTML, Tailwind CSS

Server: Gunicorn

Deployment: Render

## Feature Selection and Data Preprocessing
Before training the model, a rigorous feature selection process was carried out to ensure that only the most relevant and statistically significant features were used. This improves model performance and reduces complexity.

Chi-Square Test: This test was used to determine the relationship between categorical features (like MARITALSTATUS, EDUCATION) and the categorical target variable (Approved_Flag). Features with a p-value less than 0.05 were considered significant.

Variance Inflation Factor (VIF): To handle multicollinearity among numerical features, VIF was calculated. Features with a high VIF (greater than 6) were sequentially removed to eliminate redundancy.

ANOVA F-test: After filtering with VIF, the remaining numerical features were tested against the categorical target variable using the ANOVA F-test. This test determines if there are any statistically significant differences between the means of the groups for each feature.

This multi-step selection process ensures that the final set of features is robust and highly predictive.

## How It Works
Data Input: The user fills out a multi-step form on the web interface with their personal and financial details.

API Request: Upon submission, the front-end JavaScript sends the form data as a JSON object to the /predict endpoint on the Flask server.

Data Preprocessing: The Flask application receives the data and processes it to match the format required by the model. This includes mapping and one-hot encoding categorical features and aligning all columns to the model's training format.

Prediction: The preprocessed data is fed into the loaded xgb_best_model.pkl model, which outputs a numerical prediction (0, 1, 2, or 3).

API Response: The numerical prediction is mapped back to its original label (P1, P2, P3, or P4) and sent back to the front end as a JSON response.

Display Result: The front end dynamically displays the prediction result on the page, indicating whether the applicant is a high or low credit risk.

## Local Setup
To run this project on your local machine, follow these steps:

Clone the Repository:

git clone https://github.com/Ashish152003/Credit_Risk_Prediction.git
cd Credit_Risk_Prediction

Create a Virtual Environment (Recommended):

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install Dependencies:
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

Run the Flask Application:
Start the local development server.

python app.py

Open in Browser:
Navigate to http://127.0.0.1:5000 in your web browser to view and use the application.

## File Structure
├── app.py                 # The core Flask application logic
├── xgb_best_model.pkl     # The pre-trained XGBoost model file
├── requirements.txt       # A list of Python dependencies for the project
├── build.sh               # Build script for the Render deployment
└── templates/
    └── index.html         # The HTML file for the web interface
