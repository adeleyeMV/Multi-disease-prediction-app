import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


    
# Load model and preprocessing artifacts
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path_1 = os.path.join(script_directory, 'cancer_detection.joblib')
file_path_2 = os.path.join(script_directory, 'diabetes.joblib')
file_path_3 = os.path.join(script_directory, 'heart_disease.joblib')

# Load the file using the absolute path
artifact_1 = joblib.load(file_path_1)
artifact_2 = joblib.load(file_path_2)
artifact_3 = joblib.load(file_path_3)



# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['🩸 Diabetes Prediction',
                            '❤️ Heart Disease Prediction',
                            '🎗️ Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['ribbon', 'ribbon', 'ribbon'],
                           default_index=0)


# Diabetes Prediction Page
if selected == '🩸 Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')

    with col2:
        Glucose = st.number_input('Glucose Level')

    with col3:
        BloodPressure = st.number_input('Blood Pressure value')

    with col1:
        SkinThickness = st.number_input('Skin Thickness value')

    with col2:
        Insulin = st.number_input('Insulin Level')

    with col3:
        BMI = st.number_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.number_input('Age of the Person')


    user_input = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age,
    }

    # code for Prediction
    diab_diagnosis = ''

    

    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        X = pd.DataFrame([data])
        
        prediction = artifact_2['model'].predict(X)[0]
        decision = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
        return decision


    # Button to make prediction
    if st.button('Diabetes Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)



    

# Heart Disease Prediction Page
if selected == '❤️ Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age')

    with col2:
        sex = st.selectbox("Sex", ['','Male', 'Female'])

    with col3:
        cp = st.selectbox("Chest Pain type", ['','Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure')

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['','True', 'False'])

    with col1:
        restecg = st.selectbox("Resting Electrocardiographic results", 
                               ['', 'Normal', 
                                'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 
                                'showing probable or definite left ventricular hypertrophy by Estes criteria'])

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['', 'Yes', 'No'])

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['', 'Upsloping', 'Flat', 'Downsloping'])

    with col3:
        ca = st.selectbox('Major vessels colored by flourosopy', ['', 0,1,2,3,4])

    with col1:
        thal = st.selectbox('thal', ['', 'Fixed defect', 'Normal', 'Reversable defect'])




    # Mapping dictionaries for selectbox inputs
    sex_mapping = {
        'Male': 1,
        'Female': 0
    }

    fbs_mapping = {
        'True': 1,
        'False': 0
    }

    restecg_mapping = {
        'Normal': 0,
        'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 1,
        'showing probable or definite left ventricular hypertrophy by Estes criteria': 2
    }

    exang_mapping = {
        'Yes': 1,
        'No': 0
    }

    slope_mapping = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }

    thal_mapping = {
        'Normal': 2,
        'Fixed defect': 1,
        'Reversable defect': 3
    }
    cp_mapping = {
        'Typical angina': 0,
        'Atypical angina': 1,
        'Non-anginal pain': 2,
        'Asymptomatic': 3
    }

    # Convert selected values to corresponding numerical values
    sex_value = sex_mapping.get(sex, '')
    cp_value = cp_mapping.get(cp, '')
    fbs_value = fbs_mapping.get(fbs, '')
    restecg_value = restecg_mapping.get(restecg, '')
    exang_value = exang_mapping.get(exang, '')
    slope_value = slope_mapping.get(slope, '')
    thal_value = thal_mapping.get(thal, '')



    user_input = {
    'age': age,
    'sex': sex_value,
    'cp': cp_value,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_value,
    'restecg': restecg_value,
    'thalach': thalach,
    'exang': exang_value,
    'oldpeak': oldpeak,
    'slope': slope_value,
    'ca': ca,
    'thal': thal_value,
    }

    # code for Prediction
    heart_diagnosis = ''


    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        df = pd.DataFrame([data])
        X = pd.DataFrame(artifact_3['preprocessing'].transform(df),
                     columns=artifact_3['preprocessing'].get_feature_names_out())
        
        prediction = artifact_3['model'].predict(X)[0]
        decision = 'The person is having heart disease' if prediction == 1 else 'The person does not have any heart disease'
        return decision


    # Button to make prediction
    if st.button('Heart Disease Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)





# Breast Cancer Prediction Page
if selected == '🎗️ Breast Cancer Prediction':

    # page title
    st.title('Breast Cancer Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input('Radius Mean', min_value=0.0, max_value=50.0, step=0.1)

    with col2:
        texture_mean = st.number_input('Texture Mean', min_value=0.0, max_value=50.0, step=0.1)

    with col3:
        smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=1.0, step=0.01)

    with col1:
        compactness_mean = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, step=0.01)

    with col2:
        concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=1.0, step=0.01)

    with col3:
        symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, max_value=1.0, step=0.01)

    with col1:
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, max_value=1.0, step=0.01)

    with col2:
        radius_se = st.number_input('Radius Standard Error', min_value=0.0, max_value=5.0, step=0.01)

    with col3:
        texture_se = st.number_input('Texture Standard Error', min_value=0.0, max_value=5.0, step=0.01)

    with col1:
        smoothness_se = st.number_input('Smoothness Standard Error', min_value=0.0, max_value=0.5, step=0.01)

    with col2:
        compactness_se = st.number_input('Compactness Standard Error', min_value=0.0, max_value=0.5, step=0.01)

    with col3:
        concavity_se = st.number_input('Concavity Standard Error', min_value=0.0, max_value=0.5, step=0.01)

    with col1:
        concave_points_se = st.number_input('Concave Points Standard Error', min_value=0.0, max_value=0.5, step=0.01)

    with col2:
        symmetry_se = st.number_input('Symmetry Standard Error', min_value=0.0, max_value=0.5, step=0.01)

    with col3:
        fractal_dimension_se = st.number_input('Fractal Dimension Standard Error', min_value=0.0, max_value=0.5, step=0.01)

    with col1:
        smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0, max_value=1.0, step=0.01)

    with col2:
        compactness_worst = st.number_input('Compactness Worst', min_value=0.0, max_value=1.0, step=0.01)

    with col3:
        symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0, max_value=1.0, step=0.01)

    with col1:
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.0, max_value=1.0, step=0.01)


    user_input = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }


    # code for Prediction
    breast_cancer_diagnosis = ''


    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        df = pd.DataFrame([data])
        X = pd.DataFrame(artifact_1['preprocessing'].transform(df),
                        columns=artifact_1['cols'])
        
        prediction = artifact_1['model'].predict(X)[0]
        decision = 'Malignant (Cancerous)' if prediction == 1 else 'Benign (Non Cancerous)'
        return f"The tumor is {decision}"


    # Button to make prediction
    if st.button('Breast Cancer Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)


