import streamlit as st
import pickle
import numpy as np

# Set the title of the app
st.title('Credit Card Fraud Detection')

# Load the trained model
with open('xgb_fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scalers
with open('scaler_amount.pkl', 'rb') as f:
    scaler_amount = pickle.load(f)

with open('scaler_time.pkl', 'rb') as f:
    scaler_time = pickle.load(f)

# Input fields for 'Time' and 'Amount'
time = st.number_input('Time', min_value=0.0, format="%.2f")
amount = st.number_input('Amount', min_value=0.0, format="%.2f")

# Input fields for V1 through V28
v_features = []
for i in range(1, 29):
    value = st.number_input(f'V{i}', format="%.6f")
    v_features.append(value)

# Scale 'Time' and 'Amount'
scaled_time = scaler_time.transform([[time]])[0][0]
scaled_amount = scaler_amount.transform([[amount]])[0][0]

# Combine all features into a single list
features = [scaled_time, scaled_amount] + v_features

# When the 'Predict' button is clicked
if st.button('Predict'):
    # Convert features to a NumPy array and reshape for prediction
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)

    # Display the prediction result
    if prediction[0] == 1:
        st.error('⚠️ The transaction is predicted to be Fraudulent.')
    else:
        st.success('✅ The transaction is predicted to be Legitimate.')
