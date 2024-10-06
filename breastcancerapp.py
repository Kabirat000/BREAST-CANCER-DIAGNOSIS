import numpy as np
import streamlit as st
import pickle

# Load the saved model and scaler
scaler = pickle.load(open('scaler.sav', 'rb'))
kmeans = pickle.load(open('trained_kmeans_model.sav', 'rb'))

# Function for breast cancer prediction
def breastcancer_prediction(input_data):
    # Convert the input data into a NumPy array
    input_data_as_numpy = np.asarray(input_data)

    # Reshape the array for a single instance prediction
    input_data_reshape = input_data_as_numpy.reshape(1, -1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_reshape)

    # Make prediction using the trained clustering model
    prediction = kmeans.predict(input_data_scaled)

    # Return the diagnosis result based on the prediction
    if prediction[0] == 0:
        return 'non cancerous'
    else:
        return 'cancerous'

def main():
    # Give a title for the interface
    st.title('BREAST CANCER DIAGNOSIS Web App')
    
    # Collect inputs for 'mean' columns using Streamlit's widgets
    radius_mean = st.number_input("Enter radius (mean):", format="%.5f")
    texture_mean = st.number_input("Enter texture (mean):", format="%.5f")
    perimeter_mean = st.number_input("Enter perimeter (mean):", format="%.5f")
    area_mean = st.number_input("Enter area (mean):", format="%.5f")
    smoothness_mean = st.number_input("Enter smoothness (mean):", format="%.5f")
    compactness_mean = st.number_input("Enter compactness (mean):", format="%.5f")
    concavity_mean = st.number_input("Enter concavity (mean):", format="%.5f")
    concave_points_mean = st.number_input("Enter concave points (mean):", format="%.5f")
    symmetry_mean = st.number_input("Enter symmetry (mean):", format="%.5f")
    fractal_dimension_mean = st.number_input("Enter fractal dimension (mean):", format="%.5f")
    
    # Collect inputs for 'SE' columns
    radius_se = st.number_input("Enter radius (SE):", format="%.5f")
    texture_se = st.number_input("Enter texture (SE):", format="%.5f")
    perimeter_se = st.number_input("Enter perimeter (SE):", format="%.5f")
    area_se = st.number_input("Enter area (SE):", format="%.5f")
    smoothness_se = st.number_input("Enter smoothness (SE):", format="%.5f")
    compactness_se = st.number_input("Enter compactness (SE):", format="%.5f")
    concavity_se = st.number_input("Enter concavity (SE):", format="%.5f")
    concave_points_se = st.number_input("Enter concave points (SE):", format="%.5f")
    symmetry_se = st.number_input("Enter symmetry (SE):", format="%.5f")
    fractal_dimension_se = st.number_input("Enter fractal dimension (SE):", format="%.5f")
    
    # Collect inputs for 'Worst' columns
    radius_worst = st.number_input("Enter radius (Worst):", format="%.5f")
    texture_worst = st.number_input("Enter texture (Worst):", format="%.5f")
    perimeter_worst = st.number_input("Enter perimeter (Worst):", format="%.5f")
    area_worst = st.number_input("Enter area (Worst):", format="%.5f")
    smoothness_worst = st.number_input("Enter smoothness (Worst):", format="%.5f")
    compactness_worst = st.number_input("Enter compactness (Worst):", format="%.5f")
    concavity_worst = st.number_input("Enter concavity (Worst):", format="%.5f")
    concave_points_worst = st.number_input("Enter concave points (Worst):", format="%.5f")
    symmetry_worst = st.number_input("Enter symmetry (Worst):", format="%.5f")
    fractal_dimension_worst = st.number_input("Enter fractal dimension (Worst):", format="%.5f")

    # Create a button for prediction
    if st.button('Breast Cancer Result'):
        # Make a prediction
        diagnosis = breastcancer_prediction([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
                                             compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                                             radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, 
                                             concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                                             radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, 
                                             compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst])
        
        # Display the result
        st.success(f"The predicted diagnosis is: {diagnosis}")

if __name__ == '__main__':
    main()

    
   
