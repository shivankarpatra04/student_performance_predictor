import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

# Load preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the model
model = tf.keras.models.load_model('model.keras')

# Create a StandardScaler for GPA
gpa_scaler = StandardScaler()

# Fit the scaler with the typical GPA range
gpa_scaler.fit(np.array([0, 4]).reshape(-1, 1))

def get_letter_grade(gpa):
    # Convert numerical GPA to letter grade
    if gpa >= 3.7:
        return '+A'
    elif gpa >= 3.3:
        return 'A'
    elif gpa >= 3.0:
        return 'B+'
    elif gpa >= 2.7:
        return 'B'
    elif gpa >= 2.3:
        return 'C+'
    elif gpa >= 2.0:
        return 'C'
    elif gpa >= 1.7:
        return 'D+'
    elif gpa >= 1.3:
        return 'D'
    else:
        return 'F'

def custom_predictor(input_data):
    # Validate and preprocess input data
    for column, value in input_data.items():
        if column == 'ParentalEducation' and value[0] not in range(5):
            st.warning(f"Invalid value for ParentalEducation. Using default value 0.")
            input_data[column] = [0]
        elif column == 'ParentalSupport' and value[0] not in range(5):  
            st.warning(f"Invalid value for ParentalSupport. Using default value 0.")
            input_data[column] = [0]
    
    try:
        input_data_processed = preprocessor.transform(input_data)
    except NotFittedError:
        st.warning("Preprocessor is not fitted. Using raw input data for prediction.")
        input_data_processed = input_data.values
    except ValueError as e:
        st.error(f"Error in preprocessing: {e}")
        st.error("Using default values for prediction. Results may not be accurate.")
        input_data_processed = np.zeros((1, input_data.shape[1]))
    
    scaled_predictions = model.predict(input_data_processed)
    
    # Unscale the predictions
    unscaled_predictions = gpa_scaler.inverse_transform(scaled_predictions)
    
    return unscaled_predictions

# Streamlit app title
st.title('Student Performance Predictor')

st.subheader("Enter Student Information")

# Input fields for student information
Age = st.number_input('Age', min_value=10, max_value=100, value=18)
Gender = st.selectbox('Gender', options=['Male', 'Female'])
Ethnicity = st.selectbox('Ethnicity', options=['White', 'Black', 'Hispanic', 'Asian', 'Other'])
ParentalEducation = st.selectbox('Parental Education Level', 
                                 options=['High School', 'Some College', 'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree or higher'])
StudyTimeWeekly = st.number_input('Study Time Weekly (hours)', min_value=0.0, max_value=100.0, value=10.0)
Absences = st.number_input('Absences', min_value=0, max_value=100, value=5)
Tutoring = st.selectbox('Tutoring', options=['No', 'Yes'])
ParentalSupport = st.selectbox('Parental Support', options=['None', 'Very Low', 'Low', 'Medium', 'High'])
Extracurricular = st.selectbox('Extracurricular Activities', options=['No', 'Yes'])
Sports = st.selectbox('Sports Participation', options=['No', 'Yes'])
Music = st.selectbox('Music Participation', options=['No', 'Yes'])
Volunteering = st.selectbox('Volunteering', options=['No', 'Yes'])






# Convert categorical inputs to numerical values
gender_map = {'Male': 0, 'Female': 1}
ethnicity_map = {'White': 0, 'Black': 1, 'Hispanic': 2, 'Asian': 3, 'Other': 4}
education_map = {'High School': 0, 'Some College': 1, 'Associate\'s Degree': 2, 'Bachelor\'s Degree': 3, 'Master\'s Degree or higher':4}
support_map = {'None': 0,'Very Low':1, 'Low': 2, 'Medium': 3, 'High': 4}
binary_map = {'No': 0, 'Yes': 1}

if st.button('Predict GPA'):
    input_data = pd.DataFrame({
        'Age': [Age],
        'Gender': [gender_map[Gender]],
        'Ethnicity': [ethnicity_map[Ethnicity]],
        'ParentalEducation': [education_map[ParentalEducation]],
        'StudyTimeWeekly': [StudyTimeWeekly],
        'Absences': [Absences],
        'Tutoring': [binary_map[Tutoring]],
        'ParentalSupport': [support_map[ParentalSupport]],
        'Extracurricular': [binary_map[Extracurricular]],
        'Sports': [binary_map[Sports]],
        'Music': [binary_map[Music]],
        'Volunteering': [binary_map[Volunteering]]
    })

    predicted_gpa = custom_predictor(input_data)[0][0]
    letter_grade = get_letter_grade(predicted_gpa)
    
    st.markdown("## Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Predicted GPA")
        st.markdown(f"<h1 style='text-align: center; color: #1c6eff;'>{predicted_gpa:.2f}</h1>", unsafe_allow_html=True)

    with col2:
        st.markdown("### Corresponding Grade")
        st.markdown(f"<h1 style='text-align: center; color: #1c6eff;'>{letter_grade}</h1>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("GPA Scale Reference:")
    scale_data = {
        'Grade': ['A+', 'A', 'B', 'B+', 'C+', 'C', 'D+', 'D', 'F'],
        'GPA': ['4.0', '3.7', '3.3', '3.0', '2.7', '2.3', '2.0', '1.7', 'Below 1.7']
    }
    scale_df = pd.DataFrame(scale_data)
    st.table(scale_df)
    
    st.caption("Note: F is a failing grade")
