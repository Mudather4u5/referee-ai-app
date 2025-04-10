
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# تحميل النموذج والمقياس
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# عنوان التطبيق
st.title("نموذج الذكاء الاصطناعي للتحكيم الرياضي")

# واجهة الإدخال
st.header("أدخل بيانات المباراة:")
fouls = st.number_input("عدد المخالفات (Fouls):", min_value=0)
yellow = st.number_input("عدد الكروت الصفراء (Yellow Cards):", min_value=0)
red = st.number_input("عدد الكروت الحمراء (Red Cards):", min_value=0)
penalties = st.number_input("عدد ركلات الجزاء (Penalties Awarded):", min_value=0)
offsides = st.number_input("عدد التسللات (Offsides):", min_value=0)

# عند الضغط على زر التنبؤ
if st.button("تنبؤ بقرار الحكم"):
    input_data = pd.DataFrame([[fouls, yellow, red, penalties, offsides]],
                              columns=["Fouls", "Yellow_Cards", "Red_Cards", "Penalties_Awarded", "Offsides"])
    
    # مطابقة الأعمدة وتطبيق القياس
    input_scaled = scaler.transform(input_data)
    
    # التنبؤ
    prediction = model.predict(input_scaled)

    st.success(f"القرار المتوقع: {prediction[0]}")
