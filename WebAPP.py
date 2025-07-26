import numpy as np
import streamlit as st
import pickle
# from sklearn.preprocessing import StandardScaler
    

with open("trained_model_with_scaler.sav", "rb") as f:
    loaded_model, scaler = pickle.load(f)

# creating a function for prediction

def diabetes_prediction(input_data): 
       
# Convert input data to numpy array and reshape it
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)

# Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)

# Predict using the loaded model
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "This is not a diabetic person"
    else:
        return"This person is diabetic"
    



def main():

    #giving title to the page
    st.title("Diabetes Prediction Web APP")

    # taking input from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DiabetesPedigreeFunction = st.number_input("Diabetes")
    Age = st.text_input("Enter Your Age")
    
    
    
    # code for prediction
    diagnosis = ""

    # creating a button for prediction

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ == "__main__":
    main()
