import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

# Load the saved model
model_path = r"C:\Users\DELL\Desktop\Loan_Default_Predection\ML_MODEL\knn_model.pkl"
model = joblib.load(model_path)

# Define the list of feature columns
feature_columns = [
    'Home Ownership',
    'Annual Income',
    'Years in current job',
    'Tax Liens',
    'Number of Open Accounts',
    'Years of Credit History',
    'Maximum Open Credit',
    'Number of Credit Problems',
    'Months since last delinquent',
    'Bankruptcies',
    'Purpose',
    'Term',
    'Current Loan Amount',
    'Current Credit Balance',
    'Monthly Debt',
    'Credit Score'
]

# Function to preprocess input data
def preprocess_input(input_data):
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data)

    # Impute missing values with the mean value of each column
    imputer = SimpleImputer(strategy='mean')
    input_data_imputed = imputer.fit_transform(input_data)
    
    # Convert the imputed features back to a DataFrame
    input_data_imputed = pd.DataFrame(input_data_imputed, columns=input_data.columns)

    # Reorder columns to match training set
    input_data_imputed = input_data_imputed.reindex(columns=feature_columns, fill_value=0)

    return input_data_imputed

def main():
    # Set the title of the web app
    st.title('Loan Default Prediction')

    # Add a description
    st.write('Enter loan applicant information to predict default risk.')

    # Create input fields for each feature
    with st.form(key='loan_form'):
        col1, col2 = st.columns(2)

        with col1:
            home_ownership = st.selectbox('Home Ownership', ['Rent', 'Own Home', 'Mortgage', 'Other'])
            annual_income = st.number_input('Annual Income', min_value=0, step=1000)
            years_in_job = st.number_input('Years in Current Job', min_value=0, step=1)
            tax_liens = st.number_input('Tax Liens', min_value=0, step=1)
            num_open_accounts = st.number_input('Number of Open Accounts', min_value=0, step=1)
            years_credit_history = st.number_input('Years of Credit History', min_value=0, step=1)
            max_open_credit = st.number_input('Maximum Open Credit', min_value=0, step=1000)
            num_credit_problems = st.number_input('Number of Credit Problems', min_value=0, step=1)

        with col2:
            months_last_delinquent = st.number_input('Months Since Last Delinquent', min_value=0, step=1)
            bankruptcies = st.number_input('Bankruptcies', min_value=0, step=1)
            purpose = st.selectbox('Purpose', ['Debt Consolidation', 'Home Improvements', 'Other'])
            term = st.selectbox('Term', ['Short Term', 'Long Term'])
            current_loan_amount = st.number_input('Current Loan Amount', min_value=0, step=1000)
            current_credit_balance = st.number_input('Current Credit Balance', min_value=0, step=1000)
            monthly_debt = st.number_input('Monthly Debt', min_value=0, step=100)
            credit_score = st.number_input('Credit Score', min_value=0, max_value=850, step=1)

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Home Ownership': [home_ownership],
            'Annual Income': [annual_income],
            'Years in current job': [years_in_job],
            'Tax Liens': [tax_liens],
            'Number of Open Accounts': [num_open_accounts],
            'Years of Credit History': [years_credit_history],
            'Maximum Open Credit': [max_open_credit],
            'Number of Credit Problems': [num_credit_problems],
            'Months since last delinquent': [months_last_delinquent],
            'Bankruptcies': [bankruptcies],
            'Purpose': [purpose],
            'Term': [term],
            'Current Loan Amount': [current_loan_amount],
            'Current Credit Balance': [current_credit_balance],
            'Monthly Debt': [monthly_debt],
            'Credit Score': [credit_score]
        })

        # Preprocess the input data
        input_data_preprocessed = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(input_data_preprocessed)
        probability = model.predict_proba(input_data_preprocessed)[0][1]

        # Display the prediction results
        st.subheader('Prediction')
        st.write(f'Default Risk: {"Yes" if prediction[0] == 1 else "No"}')
        st.write(f'Probability of Default: {probability:.2f}')

if __name__ == '__main__':
    main()
