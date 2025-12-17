# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 21:20:07 2025

@author: M Henry_Leandre
"""
import numpy as np
import pickle
# loading the saved model
loaded_model = pickle.load(open('C:/Users/M Henry_Leandre/Desktop/Deploy/trained_model.sav','rb'))


input_data = (2020,6, 420, 90, 36, 86)  
# daily_usage_hours, charging_cycles, avg_charge_limit_percent,
# battery_age_months, battery_health_percent

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if prediction[0] == 0:
    print('The laptop has no overheating issue')
else:
    print('The laptop has an overheating issue')