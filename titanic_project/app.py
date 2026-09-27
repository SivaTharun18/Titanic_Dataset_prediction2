from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'titanic_best_model.pkl'
RESULTS_PATH = ROOT / 'models' / 'model_comparison.csv'
META_PATH = ROOT / 'models' / 'metadata.json'

st.set_page_config(page_title='Titanic Survival Predictor', page_icon='🚢', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

with open(META_PATH, encoding='utf-8') as f:
    metadata = json.load(f)

st.markdown('''<style>
.main {padding-top: 1rem;}
.hero {padding: 2rem; border-radius: 20px; background: linear-gradient(135deg,#0f172a,#1e3a5f); color:white; margin-bottom:1.5rem;}
.hero h1 {font-size: 2.6rem; margin-bottom: .3rem;}
.result {padding:1.5rem; border-radius:18px; background:#f8fafc; border:1px solid #dbe4ee;}
</style>''', unsafe_allow_html=True)

st.markdown(f'''<div class="hero"><h1>🚢 Titanic Survival Predictor</h1><p>Machine Learning dashboard for passenger survival prediction</p><p><b>Selected model:</b> {metadata['best_model']}</p></div>''', unsafe_allow_html=True)

left, right = st.columns([1, 1.2], gap='large')
with left:
    st.subheader('Passenger Information')
    with st.form('prediction_form'):
        pclass = st.selectbox('Passenger Class', [1, 2, 3], index=2)
        sex = st.selectbox('Sex', ['male', 'female'])
        age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        sibsp = st.number_input('Siblings / Spouses Aboard', min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input('Parents / Children Aboard', min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input('Fare', min_value=0.0, max_value=600.0, value=30.0, step=0.5)
        embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])
        submitted = st.form_submit_button('Predict Survival', use_container_width=True)

with right:
    st.subheader('Prediction Result')
    if submitted:
        model = load_model()
        input_data = pd.DataFrame([{'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}])
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        survival_probability = float(probabilities[1])
        if prediction == 1:
            st.success('Passenger is predicted to SURVIVE.')
        else:
            st.error('Passenger is predicted NOT TO SURVIVE.')
        c1, c2 = st.columns(2)
        c1.metric('Survival Probability', f'{survival_probability:.2%}')
        c2.metric('Non-survival Probability', f'{probabilities[0]:.2%}')
        st.progress(survival_probability)
        st.caption('This is a statistical model prediction, not a historical certainty.')
    else:
        st.info('Enter passenger details and click Predict Survival to view the result.')

st.divider()
st.subheader('Model Comparison')
results = load_results()
st.dataframe(results.style.format({c: '{:.3f}' for c in results.columns if c != 'Model'}), use_container_width=True, hide_index=True)
st.bar_chart(results.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']])
st.caption('Models were trained on an 80/20 stratified train-test split with preprocessing for missing values, encoding, and scaling.')
