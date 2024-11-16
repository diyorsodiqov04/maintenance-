# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder  # Ensure this is imported
from sklearn.ensemble import RandomForestClassifier

# Modelni yuklash
model_path = r'C:\Users\Public\fanlar\5-smestr\suniy\or1.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model fayli topilmadi: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Modelni yuklashda xato: {str(e)}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="centered")
st.title("Machine Predictive Maintenance Prediction")

# Foydalanuvchi kiritmalari
udi = st.text_input("UDI (masalan, 12345)", "12345")
product_id = st.text_input("Mahsulot ID (masalan, L47181)", "L47181")
failure_type = st.selectbox("Failure Type", ["L", "H"])  # Add all failure types from the dataset

air_temp = st.number_input("Air Temperature (K)", min_value=0.0, max_value=500.0)
process_temp = st.number_input("Process Temperature (K)", min_value=0.0, max_value=500.0)
rotational_speed = st.number_input("Rotational Speed (rpm)", min_value=0.0, max_value=3000.0)
torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=1000.0)
tool_wear = st.number_input("Tool Wear (min)", min_value=0.0, max_value=500.0)

# Foydalanuvchi kiritgan ma'lumotlar bilan prediksiya
if st.button("Predict Maintenance Needs"):
    # Kategoriyalarni raqamli qiymatlarga aylantirish (agar kerak bo'lsa)
    label_encoder = LabelEncoder()
    encoded_failure_type = label_encoder.fit_transform([failure_type])[0]
    
    # Ma'lumotlarni to'plamga qo'shish
    input_data = pd.DataFrame([[udi, product_id, failure_type, air_temp, process_temp, rotational_speed, torque, tool_wear]],
                              columns=["UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]", 
                                       "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
    
    # Input data uchun "Product ID"ni olib tashlash, faqat kerakli xususiyatlar qoldirilsin
    input_data = input_data.drop(["Product ID"], axis=1)  # Bu ustunni modelga yubormaymiz
    
    # Modellan oldingi xususiyatlar bilan moslashtirish
    model_features = ["UDI", "Type", "Air temperature [K]", "Process temperature [K]",
                      "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    
    # Ensure that the input data columns match the training data columns exactly
    input_data = input_data[model_features]
    
    # Predict qilish
    prediction = model.predict(input_data)  # Xatolikni oldini olish uchun faqat kerakli xususiyatlar
    st.success(f"Bashorat: {'Failure Expected' if prediction[0] == 1 else 'No Failure Expected'}")
