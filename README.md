# 📊 Obesity Prediction using Stacking Ensemble Models and Explainable AI (XAI)

## 📌 Description
This project focuses on predicting multiple levels of obesity using Machine Learning techniques. It uses a stacking ensemble approach that combines multiple models to improve prediction accuracy and robustness. Additionally, Explainable AI (XAI) techniques are applied to provide interpretable insights into the model's decisions.

---

## ❗ Problem Statement
Obesity is a major public health concern associated with diseases such as diabetes, hypertension, and cardiovascular disorders. Traditional methods for obesity prediction often rely on Body Mass Index (BMI), which does not accurately reflect an individual's health condition or fat distribution.

Two individuals with the same BMI may have different lifestyle patterns and health risks. Therefore, there is a need for a more accurate and reliable system that considers multiple factors.

This project aims to develop a multi-level obesity prediction model using a stacking ensemble approach that combines XGBoost, LightGBM, and Logistic Regression. The system also integrates Explainable AI (LIME) to provide transparent and interpretable predictions.

---

## 🎯 Objectives
- Develop a stacking-based predictive model for obesity classification  
- Improve prediction accuracy using multiple algorithms  
- Identify key factors influencing obesity  
- Provide explainable predictions using LIME  

---

## 📊 Dataset
- **Dataset:** Obesity Risk Prediction Dataset  
- **Records:** 2,111  
- **Features:** 17  

### Attributes:
- Age, Height, Weight  
- Eating habits  
- Physical activity  
- Lifestyle behaviors  

---

## ⚙️ Methodology

### 🔹 Step 1: Data Preparation
- Load dataset  
- Handle missing values  
- Encode categorical variables  
- Normalize numerical features  

### 🔹 Step 2: Model Training
Train multiple base models:
- XGBoost  
- LightGBM  

### 🔹 Step 3: Stacking Ensemble
- Generate predictions from base models  
- Use predictions as input features  
- Train Logistic Regression as meta-model  

### 🔹 Step 4: Model Evaluation
- Evaluate models using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  

### 🔹 Step 5: Explainable AI (XAI)
- Apply LIME  
- Identify important features  
- Explain predictions in human-understandable form  

---

## 🤖 Algorithms Used

- XGBoost  
- LightGBM  
- Logistic Regression (Meta-model)  

---

## 📈 Results

| Model               | Accuracy |
|--------------------|---------|
| XGBoost            | 95.27%  |
| Decision Tree      | 85.00%  |
| Logistic Regression| 81.79%  |

👉 The stacking model improves overall performance and provides reliable predictions. :contentReference[oaicite:0]{index=0}

---

## 📦 Requirements

Install dependencies:
