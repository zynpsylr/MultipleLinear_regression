import streamlit as st
import numpy as np
import joblib

st.title ("💰 Reklam Bütçesi ile Satış Tahmini Uygulaması")

TVAdBudget = st.number_input("📺 TV reklamı için harcanan tutarı girin (K$):",min_value=0, max_value=900, value=1, step=1)
RadioAdBudget= st.number_input("📻 Radyo reklamı için harcanan tutarı girin (K$) :",min_value=0, max_value=800, value=1, step=1)
NewspaperAdBudget = st.number_input("📰 Gazete reklamı için harcanan tutarı girin (K$):",min_value=0, max_value=800, value=1, step=1)

# Butona basıldığında tahmin edilecek
if st.button("Tahmin et"):
    # Modeli yükle 
    model= joblib.load('multiplelinear_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Kullanıcıdan alınan veriyi array'e dönüştür
    input_data = np.array([[TVAdBudget, RadioAdBudget, NewspaperAdBudget]])

    # Veriyi ölçeklendir
    input_scaled = scaler.transform(input_data)

    # Tahmin yap
    prediction = model.predict(input_scaled) 
    
    # Sonucu göster
    st.success(f"Tahmini satış geliri: {prediction[0]:,.2f} (M$) ")
  
