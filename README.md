# 🔍 Credit Risk Prediction Web Application

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-006600?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" />
</p>

> A web application that predicts **credit risk** based on personal and financial data using a trained **XGBoost ML model**. Built with a clean multi-step form on the front end and a robust Flask API on the backend.

🎯 **[Live Demo](https://credit-risk-predictor-02fj.onrender.com/)**

---

## 🚀 Features

- ✅ **Multi-Step Form:** Intuitive and grouped inputs for a smoother experience  
- ⚡ **Real-Time Prediction:** Instantly returns predictions using a Flask API  
- 🎯 **XGBoost Model:** High-performing classifier for credit risk prediction  
- 💻 **Clean & Responsive UI:** Built with Tailwind CSS for all screen sizes  
- 🌐 **Deployed on Render:** Live and accessible from anywhere  

---

## 🧠 Tech Stack

| Layer         | Tools Used                                  |
|---------------|---------------------------------------------|
| **Backend**   | Python, Flask                               |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, NumPy     |
| **Frontend**  | HTML, Tailwind CSS                          |
| **Server**    | Gunicorn                                    |
| **Deployment**| Render                                      |

---

## 📊 Feature Selection & Preprocessing

Robust preprocessing steps were performed to enhance model performance:

- **Chi-Square Test:** Used for categorical features (`EDUCATION`, `MARITALSTATUS`) with the `Approved_Flag` target (p < 0.05).
- **VIF (Variance Inflation Factor):** Removed numerical features with multicollinearity (VIF > 6).
- **ANOVA F-test:** Selected statistically significant numerical features.

> This pipeline ensures minimal redundancy and maximum relevance.

---

## ⚙️ How It Works

1. **🧾 User Input:** Multi-step form collects user data
2. **📤 API Request:** Data sent to `/predict` Flask endpoint as JSON
3. **🧪 Preprocessing:** Data is encoded/mapped to model format
4. **🤖 Prediction:** Model returns a class (0–3)
5. **🔁 Response Mapping:** Output mapped to risk labels (`P1`, `P2`, `P3`, `P4`)
6. **📺 Result Display:** Prediction shown dynamically on frontend

---

## 🧪 Local Setup

### 📥 Clone the Repository

```bash
git clone https://github.com/Ashish152003/Credit_Risk_Prediction.git
cd Credit_Risk_Prediction
```

### 🧰 Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the Application

```bash
python app.py
```

Then open your browser at:  
👉 **http://127.0.0.1:5000**

---

## 📁 File Structure

```
Credit_Risk_Prediction/
├── app.py                 # Flask backend logic
├── xgb_best_model.pkl     # Trained ML model
├── requirements.txt       # Project dependencies
├── build.sh               # Deployment script for Render
└── templates/
    └── index.html         # Web interface (multi-step form)
```

---

## 🙌 Acknowledgements

- Tailwind CSS for the sleek UI
- XGBoost and Scikit-learn for powerful ML capabilities
- Render for seamless deployment

