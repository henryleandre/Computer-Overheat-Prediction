# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 23:33:31 2025

@author: M Henry_Leandre
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 21:38:19 2025

@author: M Henry_Leandre
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


def computer_prediction(input_data):
    
  
# daily_usage_hours, charging_cycles, avg_charge_limit_percent,
# battery_age_months, battery_health_percent

   # Changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # Reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

  prediction = loaded_model.predict(input_data_reshaped)
  print(prediction)
  if prediction[0] == 0:
    return'The laptop has no overheating issue'
  else:
    return'The laptop has an overheating issue'
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image:
                linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                url("https://images.unsplash.com/photo-1517336714731-489689fd1ca8");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Make all text white */
        .stApp, .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp h5, .stApp h6, .stApp p,
        .stApp label {
            color: white;
        }

        /* Input fields styling */
        input, textarea, select {
            background-color: #1e1e1e !important;
            color: white !important;
            border-radius: 8px;
        }

        /* Button styling */
        button {
            background-color: #ff4b4b !important;
            color: white !important;
            border-radius: 10px;
            font-weight: bold;
        }

        button:hover {
            background-color: #ff1f1f !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )     
    
def main():
      add_bg_from_url()
      #    create title
      
      st.title("💻 Laptop Overheating Prediction App")
      st.markdown(
        "This application predicts **laptop overheating issues** "
        "based on usage and battery health parameters using a "
        "**machine learning model**."
      )
      st.header("Enter Laptop Parameters")

      # User inputs
      model_year = st.number_input("Model Year", min_value=2010, max_value=2030, value=2021)
      daily_usage_hours = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=6.0)
      charging_cycles = st.number_input("Charging Cycles", min_value=0, value=400)
      avg_charge_limit_percent = st.number_input("Average Charge Limit (%)", min_value=50, max_value=100, value=90)
      battery_health_percent = st.number_input("Battery Health (%)", min_value=0, max_value=100, value=85)
      battery_age_months = st.number_input("Battery Age (Months)", min_value=0, value=36)
   # code for prediction 
      diagnosis=''
      # Prediction button
      if st.button("🔍Predict Overheating Issue"):

         diagnosis = computer_prediction([[model_year,
                             daily_usage_hours,
                             charging_cycles,
                             avg_charge_limit_percent,
                             battery_health_percent,
                             battery_age_months]])

      st.success(diagnosis)
      
    
if __name__=='__main__':
    main()

