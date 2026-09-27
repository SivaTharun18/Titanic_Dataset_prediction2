# Titanic Survival Prediction — Streamlit Dashboard

A machine learning project that predicts Titanic passenger survival using passenger class, sex, age, family information, fare, and embarkation port.

## Features
- Five classification models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and ANN (MLP)
- Automatic preprocessing for missing values and categorical features
- Model comparison using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Interactive Streamlit prediction dashboard
- Best model selected using F1 Score and saved with Joblib

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deployment
Deploy this repository through Streamlit Community Cloud. Set the main file to `app.py`.
