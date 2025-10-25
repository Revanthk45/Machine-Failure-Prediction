import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import sklearn

# Set page config
st.set_page_config(
    page_title="Machine Learning Predictions",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("Machine Learning Prediction Models")

# Sidebar
st.sidebar.header("Select Model")
model_choice = st.sidebar.radio(
    "Choose the prediction model:",
    ["Device Failure Prediction", "Machine Failure Prediction"]
)

if model_choice == "Device Failure Prediction":
    st.header("Device Failure Prediction")
    
    # Input features for device failure
    st.subheader("Enter Device Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        sensor1 = st.number_input("Sensor 1", min_value=0.0, max_value=100.0, value=50.0)
        sensor2 = st.number_input("Sensor 2", min_value=0.0, max_value=100.0, value=50.0)
        sensor3 = st.number_input("Sensor 3", min_value=0.0, max_value=100.0, value=50.0)
        
    with col2:
        sensor4 = st.number_input("Sensor 4", min_value=0.0, max_value=100.0, value=50.0)
        sensor5 = st.number_input("Sensor 5", min_value=0.0, max_value=100.0, value=50.0)
        
    if st.button("Predict Device Failure"):
        try:
            # Load the model
            model = load('device_failure_model.joblib')
            
            # Make prediction
            features = np.array([[sensor1, sensor2, sensor3, sensor4, sensor5]])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Show results
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error("⚠️ Device Failure Predicted!")
            else:
                st.success("✅ Device is predicted to work normally")
                
            st.info(f"Probability of failure: {probability:.2%}")
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

else:
    st.header("Machine Failure Prediction")
    
    # Input features for machine failure
    st.subheader("Enter Machine Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        type_choices = ['L', 'M', 'H']
        machine_type = st.selectbox("Machine Type", type_choices)
        air_temperature = st.number_input("Air Temperature", min_value=0.0, max_value=100.0, value=25.0)
        process_temperature = st.number_input("Process Temperature", min_value=0.0, max_value=100.0, value=35.0)
        
    with col2:
        rotational_speed = st.number_input("Rotational Speed", min_value=0, max_value=3000, value=1500)
        torque = st.number_input("Torque", min_value=0.0, max_value=100.0, value=40.0)
        tool_wear = st.number_input("Tool Wear", min_value=0, max_value=300, value=100)
        
    with col3:
        twf = st.checkbox("Tool Wear Failure (TWF)")
        hdf = st.checkbox("Heat Dissipation Failure (HDF)")
        pwf = st.checkbox("Power Failure (PWF)")
        osf = st.checkbox("Overstrain Failure (OSF)")
        rnf = st.checkbox("Random Failures (RNF)")
        
    if st.button("Predict Machine Failure"):
        try:
            # Load the model
            model = load('machine_failure_model.joblib')
            
            # Prepare features
            type_encoded = [1 if machine_type == t else 0 for t in ['H', 'L', 'M']]
            features = np.array([[
                air_temperature, process_temperature, rotational_speed,
                torque, tool_wear, twf, hdf, pwf, osf, rnf
            ] + type_encoded])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Show results
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error("⚠️ Machine Failure Predicted!")
            else:
                st.success("✅ Machine is predicted to work normally")
                
            st.info(f"Probability of failure: {probability:.2%}")
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Machine Learning Models")