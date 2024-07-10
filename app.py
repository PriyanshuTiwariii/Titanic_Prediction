import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Create the Streamlit app
st.title('Titanic Survival Prediction')

# Input fields for the features
pclass = st.selectbox('Pclass', [1, 2, 3])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('SibSp', 0, 10, 0)
parch = st.number_input('Parch', 0, 10, 0)
fare = st.slider('Fare', 0, 500, 50)
male = st.selectbox('Sex', [1, 0])
embark = st.selectbox('Embarked', [ 'Q', 'S','None'])

# Preprocess the inputs
# sex = 0 if sex == 'male' else 1
# embark_C = 1 if embark == 'C' else 0
Q = 1 if embark == 'Q' else 0
S = 1 if embark == 'S' else 0
if embark == 'None':
    Q=0
    S=0





# Create the input dataframe
input_df = pd.DataFrame([[pclass, age, sibsp, parch, fare, male, Q , S]],
                         columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q' ,'S'])

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Display the result
if prediction == 1:
    st.markdown('<h3 style="color:blue; font-size:30px;">The passenger is likely to survive.</h3>', unsafe_allow_html=True)
else:
    st.markdown('<h3 style="color:red; font-size:30px;">The passenger is unlikely to survive.</h3>', unsafe_allow_html=True)

st.markdown(f'<h4 style="font-size:30 px;">Prediction probability: {prediction_proba[1]:.2f}</h4>', unsafe_allow_html=True)