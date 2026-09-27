from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'Titanic-Dataset.csv'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X, y = df[features], df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorical = ['Sex', 'Embarked']
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical),
])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'ANN (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=42),
}

results = []
trained = {}
for name, estimator in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1 Score': f1_score(y_test, pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, prob),
    })
    trained[name] = pipe

results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False).reset_index(drop=True)
best_name = results_df.iloc[0]['Model']
joblib.dump(trained[best_name], MODEL_DIR / 'titanic_best_model.pkl')
results_df.to_csv(MODEL_DIR / 'model_comparison.csv', index=False)
with open(MODEL_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump({'best_model': best_name, 'features': features}, f, indent=2)
print(results_df.to_string(index=False))
print(f'Best model: {best_name}')
