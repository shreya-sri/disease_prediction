"""
@author: shreyasri
"""

import streamlit as st
from joblib import load
import pandas as pd

#load the random forest classifier created
classifier = load('classifier_weights.joblib')

st.title('Disease Prediction Questionare')

def val_style(text):
    html_temp = f"""
    <div style = "background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)); padding: 10px 10px 10px 10px; border-radius: 20px; "> 
    <p style = " font: color:white; font-size:30px; text_align:center;"> {text} </p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

#get feature names
features = classifier.feature_names


result = {}
temp = {}
y_test = []

#Create form to get user input
form = st.form(key='symptoms')
form.text('Please answer yes or no for the following')
for feature in features:
    result[feature] = form.selectbox("Do you have " + " ".join(w for w in feature.split("_")) + " ?", ['No', 'Yes'], key=feature)
submit = form.form_submit_button('Submit')

#When the submit button is clicked
if submit:
    #create test data from user input
    #The value is 1 if user answers yes, 0 otherwise
    for feature in result.keys():
        if result[feature] == "Yes":
            temp[feature] = 1
            print(feature)
        else:
            temp[feature] = 0
    y_test.append(temp)
    y_test = pd.DataFrame.from_dict(y_test)
    #Generate prediction
    y_pred = classifier.predict(y_test)
    val_style("You might have " + y_pred[0])
    

            

