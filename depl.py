import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load the pre-trained Logistic Regression model
logistic_regression_model = joblib.load('logistic_regression_model.pkl')

# Streamlit app
def main():
    st.title('Depression Detection App')

    # Collect user input
    user_input = {}
    for question in [
        "Are you in stress?",
        "Do you feel lonely?",
        "Do you have friends?",
        "Have you recently broken up?",
        "Do you feel like going outside?",
        "Are you employed?",
        "Do you feel emotionally sensitive recently?",
        "Do you have trouble concentrating?",
        "Do you feel insomnia?",
        "Have you ever gaslit yourself?",
        "Have you faced something traumatised recently?",
        "Are you struggling with something?",
        "Are you too self-conscious?",
        "Have you lost your appetite?",
        "Have you felt suicidal?"
    ]:
        user_input[question] = st.checkbox(question)

    if st.button('Predict'):
        # Create a DataFrame from the user input
        user_data = pd.DataFrame(user_input, index=[0])

        # Use the Logistic Regression model to make predictions
        prediction = logistic_regression_model.predict(user_data)

        # Display the prediction
        st.write(f"The predicted depression label is: {prediction[0]}")

if __name__ == '__main__':
    main()
