# -*- coding: utf-8 -*-
"""
Created on Mon May 12 18:16:38 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:35:44 2025

@author: LENOVO
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

#Loading the saved mod3els

diabetes_model= pickle.load(open('diabetes_model.sav','rb'))

heart_disease_model= pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model= pickle.load(open('parkinsons_model.sav','rb'))

breast_cancer_model=pickle.load(open('breast_cancer_model.sav','rb'))

#side bar(option menu)

with st.sidebar:
    
    selected=option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinsons Prediction','Breast Cancer Prediction'],
                         icons = ['capsule','activity','person', 'person-plus'],
                         default_index=0)
    
#Diabetes prediction page
if (selected=='Diabetes Prediction'):
    #Page title
    st.title('Diabetes Prediction using ML')
    
    # Input columns
    scaler = pickle.load(open('scaler.sav', 'rb'))

    # Input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    
    with col2:
        Glucose = st.text_input("Glucose Level")
    
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    
    with col2:
        Insulin = st.text_input("Insulin Level")
    
    with col3:
        BMI = st.text_input("BMI Value")
    
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    
    with col2:
        Age = st.text_input("Age of the Person")
    
    # Prediction
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            input_data = [
                int(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
            ]
    
            input_data_np = np.asarray(input_data).reshape(1, -1)
    
            # Apply same scaler used during training
            std_input = scaler.transform(input_data_np)
    
            prediction = diabetes_model.predict(std_input)
    
            diab_diagnosis = 'The Person is Diabetic' if prediction[0] == 1 else 'The Person is not Diabetic'
    
        except Exception as e:
            diab_diagnosis = f'Error: {e}'
    
    st.success(diab_diagnosis)

    
    
  #----------------------------------------------------------------------------  
    
    
if (selected=='Heart Disease Prediction'):
    #Page title
    st.title('Heart Disease Prediction using ML')
    
    
    #columns
    col1, col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('CP value')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure value')
        
    with col2:
        chol = st.text_input('Serum cholesterol in mg/dL')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL')
    
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum heart rate')
        
    with col3:
        exang = st.text_input("Excercise Induced Angina")
        
    with col1:
        oldpeak = st.text_input("ST Depression")
    
    with col2:
        slope = st.text_input('Slope of Peak Excercise')
        
    with col3:
        ca = st.text_input('Number of major vessels')
        
    with col1:
        thal = st.text_input('Thalassemia(0 = normal)')
        
    
    age = float(age) if age != '' else 0.0
    sex = int(sex) if sex != '' else 0
    cp = int(cp) if cp != '' else 0
    trestbps = float(trestbps) if trestbps != '' else 0.0
    chol = float(chol) if chol != '' else 0.0
    fbs = int(fbs) if fbs != '' else 0
    restecg = int(restecg) if restecg != '' else 0
    thalach = float(thalach) if thalach != '' else 0.0
    exang = int(exang) if exang != '' else 0
    oldpeak = float(oldpeak) if oldpeak != '' else 0.0
    slope = int(slope) if slope != '' else 0
    ca = int(ca) if ca != '' else 0
    thal = int(thal) if thal != '' else 0
        
    heart_diagnosis = ''
    
    if st.button('Heart Test Result'):
        heart_prediction =heart_disease_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if heart_prediction[0] == 0:
            heart_diagnosis = 'The Person does not have a Heart Disease'
        else:
            heart_diagnosis = 'The Person has Heart Disease'
            
    st.success(heart_diagnosis)
        
    
    


        
    
        
    
    
if (selected=='Parkinsons Prediction'):
    #Page title
    st.title('Parkinsons Prediction using ML')
    
    col1, col2, col3 = st.columns(3)

# Create input fields for the different features with detailed descriptions
    with col1:
        MDVP_Fo = st.text_input('Fundamental Frequency (MDVP:Fo) in Hz')
    
    with col2:
        MDVP_Fhi = st.text_input('Maximum Fundamental Frequency (MDVP:Fhi) in Hz')
    
    with col3:
        MDVP_Flo = st.text_input('Minimum Fundamental Frequency (MDVP:Flo) in Hz')
    
    with col1:
        MDVP_Jitter_percent = st.text_input('Jitter Percentage (MDVP:Jitter%)')
    
    with col2:
        MDVP_Jitter_Abs = st.text_input('Absolute Jitter (MDVP:Jitter(Abs))')
    
    with col3:
        MDVP_RAP = st.text_input('Relative Average Perturbation (MDVP:RAP)')
    
    with col1:
        MDVP_PPQ = st.text_input('Pitch Perturbation Quotient (MDVP:PPQ)')
    
    with col2:
        Jitter_DDP = st.text_input('Difference of Successive Jitters (Jitter:DDP)')
    
    with col3:
        MDVP_Shim = st.text_input('Amplitude Variation (MDVP:Shimmer)')
    
    with col1:
        MDVP_Shim_dB = st.text_input('Shimmer in Decibels (MDVP:Shimmer(dB))')
    
    with col2:
        Shimmer_APQ3 = st.text_input('Amplitude Perturbation Quotient 3 (Shimmer:APQ3)')
    
    with col3:
        Shimmer_APQ5 = st.text_input('Amplitude Perturbation Quotient 5 (Shimmer:APQ5)')
    
    with col1:
        MDVP_APQ = st.text_input('Average Amplitude Perturbation Quotient (MDVP:APQ)')
    
    with col2:
        Shimmer_DDA = st.text_input('Differential Shimmer Amplitude (Shimmer:DDA)')
    
    with col3:
        NHR = st.text_input('Noise-to-Harmonics Ratio (NHR)')
    
    with col1:
        HNR = st.text_input('Harmonics-to-Noise Ratio (HNR)')
        
    
    with col2:
        RPDE = st.text_input('Recurrence Period Density Entropy (RPDE)')
    
    with col3:
        DFA = st.text_input('Detrended Fluctuation Analysis (DFA)')
    
    with col1:
        spread1 = st.text_input('Spread 1')
    
    with col2:
        spread2 = st.text_input('Spread 2')
    
    with col3:
        D2 = st.text_input('D2')
    
    with col1:
        PPE = st.text_input('Pitch Period Entropy (PPE)')
    
    # Create a prediction button
    parkinson_diagnosis = ''
    
    # Code for prediction
    if st.button('Parkinson\'s Disease Test Result'):
        # Convert inputs to numeric values for prediction
        try:
            MDVP_Fo = float(MDVP_Fo) if MDVP_Fo != '' else 0.0
            MDVP_Fhi = float(MDVP_Fhi) if MDVP_Fhi != '' else 0.0
            MDVP_Flo = float(MDVP_Flo) if MDVP_Flo != '' else 0.0
            MDVP_Jitter_percent = float(MDVP_Jitter_percent) if MDVP_Jitter_percent != '' else 0.0
            MDVP_Jitter_Abs = float(MDVP_Jitter_Abs) if MDVP_Jitter_Abs != '' else 0.0
            MDVP_RAP = float(MDVP_RAP) if MDVP_RAP != '' else 0.0
            MDVP_PPQ = float(MDVP_PPQ) if MDVP_PPQ != '' else 0.0
            Jitter_DDP = float(Jitter_DDP) if Jitter_DDP != '' else 0.0
            MDVP_Shim = float(MDVP_Shim) if MDVP_Shim != '' else 0.0
            MDVP_Shim_dB = float(MDVP_Shim_dB) if MDVP_Shim_dB != '' else 0.0
            Shimmer_APQ3 = float(Shimmer_APQ3) if Shimmer_APQ3 != '' else 0.0
            Shimmer_APQ5 = float(Shimmer_APQ5) if Shimmer_APQ5 != '' else 0.0
            MDVP_APQ = float(MDVP_APQ) if MDVP_APQ != '' else 0.0
            Shimmer_DDA = float(Shimmer_DDA) if Shimmer_DDA != '' else 0.0
            NHR = float(NHR) if NHR != '' else 0.0
            HNR = float(HNR) if HNR != '' else 0.0
            
            RPDE = float(RPDE) if RPDE != '' else 0.0
            DFA = float(DFA) if DFA != '' else 0.0
            spread1 = float(spread1) if spread1 != '' else 0.0
            spread2 = float(spread2) if spread2 != '' else 0.0
            D2 = float(D2) if D2 != '' else 0.0
            PPE = float(PPE) if PPE != '' else 0.0
    
            # Assuming the model is already loaded (parkinsons_model)
            parkinson_prediction = parkinsons_model.predict([[
                MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP,
                MDVP_PPQ, Jitter_DDP, MDVP_Shim, MDVP_Shim_dB, Shimmer_APQ3, Shimmer_APQ5,
                MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
            ]])
    
            # Provide the prediction result
            if parkinson_prediction[0] == 0:
                parkinson_diagnosis = 'The Person does not have Parkinson\'s Disease'
            else:
                parkinson_diagnosis = 'The Person has Parkinson\'s Disease'
    
        except ValueError:
            parkinson_diagnosis = 'Please enter valid numerical values for all inputs.'
    
    st.success(parkinson_diagnosis)
    
if (selected=='Breast Cancer Prediction'):
    #Page title
    st.title('Breast Cancer Prediction using ML')
    
    col1, col2, col3 = st.columns(3)

# Input fields for the new dataset
    with col1:
        mean_radius = st.text_input('Mean Radius')
    
    with col2:
        mean_texture = st.text_input('Mean Texture')
    
    with col3:
        mean_perimeter = st.text_input('Mean Perimeter')
    
    with col1:
        mean_area = st.text_input('Mean Area')
    
    with col2:
        mean_smoothness = st.text_input('Mean Smoothness')
    
    with col3:
        mean_compactness = st.text_input('Mean Compactness')
    
    with col1:
        mean_concavity = st.text_input('Mean Concavity')
    
    with col2:
        mean_concave_points = st.text_input('Mean Concave Points')
    
    with col3:
        mean_symmetry = st.text_input('Mean Symmetry')
    
    with col1:
        mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
    
    with col2:
        radius_error = st.text_input('Radius Error')
    
    with col3:
        texture_error = st.text_input('Texture Error')
    
    with col1:
        perimeter_error = st.text_input('Perimeter Error')
    
    with col2:
        area_error = st.text_input('Area Error')
    
    with col3:
        smoothness_error = st.text_input('Smoothness Error')
    
    with col1:
        compactness_error = st.text_input('Compactness Error')
    
    with col2:
        concavity_error = st.text_input('Concavity Error')
    
    with col3:
        concave_points_error = st.text_input('Concave Points Error')
    
    with col1:
        symmetry_error = st.text_input('Symmetry Error')
    
    with col2:
        fractal_dimension_error = st.text_input('Fractal Dimension Error')
    
    with col3:
        worst_radius = st.text_input('Worst Radius')
    
    with col1:
        worst_texture = st.text_input('Worst Texture')
    
    with col2:
        worst_perimeter = st.text_input('Worst Perimeter')
    
    with col3:
        worst_area = st.text_input('Worst Area')
    
    with col1:
        worst_smoothness = st.text_input('Worst Smoothness')
    
    with col2:
        worst_compactness = st.text_input('Worst Compactness')
    
    with col3:
        worst_concavity = st.text_input('Worst Concavity')
    
    with col1:
        worst_concave_points = st.text_input('Worst Concave Points')
    
    with col2:
        worst_symmetry = st.text_input('Worst Symmetry')
    
    with col3:
        worst_fractal_dimension = st.text_input('Worst Fractal Dimension')
    
    # Prediction
    diagnosis = ''
    
    if st.button("Breast Cancer Prediction Result"):
        try:
            # Convert inputs to float
            input_data = [
                mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
                radius_error, texture_error, perimeter_error, area_error, smoothness_error,
                compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
                worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
            ]
    
            input_data = [float(i) if i != '' else 0.0 for i in input_data]
    
            # Predict using your model (assumed loaded as `breast_cancer_model`)
            prediction = breast_cancer_model.predict([input_data])
    
            if prediction[0] == 0:
                diagnosis = "The tumor is Malignant"
            else:
                diagnosis = "The tumor is Benign"
    
        except ValueError:
            diagnosis = "Please enter valid numerical values for all inputs."
    
    st.success(diagnosis)

    
