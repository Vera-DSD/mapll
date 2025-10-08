import streamlit as st
import joblib
import numpy as np
import warnings
import logging
import pandas as pd

warnings.filterwarnings('ignore')

logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    return model

model = load_model()

def preprocess_input(age, sex, chest_pain, resting_bp, cholesterol,
                     fasting_bs, resting_ecg, max_hr, exercise_angina,
                     oldpeak, st_slope):
    
    processed_sex = {'Male': 1, 'Female': 0}[sex]
    processed_chest_pain = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}[chest_pain]
    processed_resting_ecg = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[resting_ecg]
    processed_exercise_angina = {'Yes': 1, 'No': 0}[exercise_angina]
    processed_st_slope = {'Up': 0, 'Flat': 1, 'Down': 2}[st_slope]
    processed_fasting_bs = 1 if fasting_bs == 'Yes' else 0
    
   
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [processed_sex],
        'ChestPainType': [processed_chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [processed_fasting_bs],
        'RestingECG': [processed_resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [processed_exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [processed_st_slope]
    })
    
    return input_data
    
   
    processed_fasting_bs = 1 if fasting_bs == 'Yes' else 0
    
    input_array = np.array([age, processed_sex, processed_chest_pain, resting_bp, cholesterol,
                           processed_fasting_bs, processed_resting_ecg, max_hr, processed_exercise_angina,
                           float(oldpeak), processed_st_slope]).reshape(1, -1)
    return input_array

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Predictor")
st.markdown("Приложение для оценки вероятности сердечно-сосудистых заболеваний на основе медицинских показателей.")

with st.form(key='patient_data'):
    st.subheader("Медицинские показатели пациента")
    
    # Размещаем элементы управления в две колонки
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Возраст", min_value=18, value=50)
        sex = st.selectbox("Пол", ["Male", "Female"])
        chest_pain = st.selectbox("Тип боли в груди", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.slider("Артериальное давление покоя", min_value=80, max_value=200, value=120)
        cholesterol = st.slider("Холестерин", min_value=100, max_value=400, value=200)
        
    with col2:
        fasting_bs = st.radio("Повышенный уровень сахара крови натощак", ['Yes', 'No'])
        resting_ecg = st.selectbox("Электрокардиограмма покоя", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.slider("Максимальная частота сердечных сокращений", min_value=60, max_value=220, value=150)
        exercise_angina = st.radio("Ангинина при физической нагрузке", ['Yes', 'No'])
        oldpeak = st.number_input("Депрессия ST сегмента", min_value=0.0, step=0.1, format="%f")
        st_slope = st.selectbox("Наклон пиковых сегментов ST", ["Up", "Flat", "Down"])

    # Кнопка отправки формы ответа 
    submitted = st.form_submit_button("Предсказать риск заболевания")

if submitted:
    data = preprocess_input(age, sex, chest_pain, resting_bp, cholesterol,
                            fasting_bs, resting_ecg, max_hr, exercise_angina,
                            oldpeak, st_slope)
    result = model.predict(data)
    if result[0]:
        st.error("Высокий риск развития сердечно-сосудистого заболевания 🩺")
    else:
        st.success("Низкий риск ❤️")



