# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_years_in_job

# Define the preprocessing function
def preprocess_years_in_job(X):
    X = X.copy()
    # Check and convert to string type if necessary
    if X['Years in current job'].dtype != 'object':
        X['Years in current job'] = X['Years in current job'].astype(str)
    # Handle possible NaN values
    X['Years in current job'] = X['Years in current job'].fillna('')
    X['Years in current job'] = X['Years in current job'].str.extract(r'(\d+)').astype(float)
    return X

# Load the saved model
model_path = r"C:\Users\DELL\Desktop\Loan_Default_Predection\ML_MODEL\knn_model.pkl"
model = joblib.load(model_path)

# Define feature names
num_features = [
    'Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts', 
    'Years of Credit History', 'Maximum Open Credit', 'Number of Credit Problems', 
    'Months since last delinquent', 'Bankruptcies', 'Current Loan Amount', 
    'Current Credit Balance', 'Monthly Debt', 'Credit Score'
]
categorical_columns = ['Home Ownership', 'Purpose', 'Term']

def main():
    # Set the title of the web app
    st.title('Loan Default Prediction')

    # Add a description
    st.write('Enter customer information to predict loan default.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Customer Information')

        # Add input fields for numerical features
        annual_income = st.number_input('Annual Income', min_value=0)
        years_in_current_job = st.slider('Years in Current Job', 0, 30, 5)
        tax_liens = st.slider('Tax Liens', 0, 10, 0)
        open_accounts = st.slider('Number of Open Accounts', 0, 50, 10)
        credit_history = st.slider('Years of Credit History', 0.0, 50.0, 10.0)
        max_credit = st.number_input('Maximum Open Credit', min_value=0)
        credit_problems = st.slider('Number of Credit Problems', 0, 10, 0)
        months_since_last_delinquent = st.slider('Months Since Last Delinquent', 0, 100, 0)
        bankruptcies = st.slider('Bankruptcies', 0, 10, 0)
        current_loan_amount = st.number_input('Current Loan Amount', min_value=0)
        current_credit_balance = st.number_input('Current Credit Balance', min_value=0)
        monthly_debt = st.number_input('Monthly Debt', min_value=0)
        credit_score = st.slider('Credit Score', 0, 1000, 500)

        # Add input fields for categorical features
        home_ownership = st.selectbox('Home Ownership', ['Have Mortgage', 'Home Mortgage', 'Own Home', 'Rent'])
        purpose = st.selectbox('Purpose', [
            'business loan', 'buy a car', 'buy house', 'debt consolidation',
            'educational expenses', 'home improvements', 'major purchase', 
            'medical bills', 'moving', 'other', 'small business', 
            'take a trip', 'vacation', 'wedding'
        ])
        term = st.selectbox('Term', ['Short Term', 'Long Term'])

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Annual Income': [annual_income],
        'Years in current job': [years_in_current_job],
        'Tax Liens': [tax_liens],
        'Number of Open Accounts': [open_accounts],
        'Years of Credit History': [credit_history],
        'Maximum Open Credit': [max_credit],
        'Number of Credit Problems': [credit_problems],
        'Months since last delinquent': [months_since_last_delinquent],
        'Bankruptcies': [bankruptcies],
        'Home Ownership': [home_ownership],
        'Purpose': [purpose],
        'Term': [term],
        'Current Loan Amount': [current_loan_amount],
        'Current Credit Balance': [current_credit_balance],
        'Monthly Debt': [monthly_debt],
        'Credit Score': [credit_score]
    })

    # Preprocess the input data
    input_data = preprocess_years_in_job(input_data)

    # Ensure all required columns are present and in the correct order
    for col in categorical_columns:
        if col not in input_data.columns:
            input_data[col] = ""

    # Ensure the columns are in the correct order
    input_data = input_data[num_features + categorical_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            # Transform input data
            input_data_transformed = model.named_steps['preprocessor'].transform(input_data)
            input_data_df = pd.DataFrame(input_data_transformed, columns=model.named_steps['preprocessor'].get_feature_names_out())

            # Make prediction
            prediction = model.named_steps['classifier'].predict(input_data_df)
            probability = model.named_steps['classifier'].predict_proba(input_data_df)[0][1]
            
            st.write(f'Prediction: {"Will Default" if prediction[0] == 1 else "Will Not Default"}')
            st.write(f'Probability of Default: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot Pass/Fail probability
            sns.barplot(x=['Will Not Default', 'Will Default'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
            axes[0].set_title('Loan Default Probability')
            axes[0].set_ylabel('Probability')

            # Plot Credit Score distribution
            sns.histplot(input_data['Credit Score'], kde=True, ax=axes[1])
            axes[1].set_title('Credit Score Distribution')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.error("This customer is likely to default on the loan. Consider further evaluation.")
            else:
                st.success("This customer is not likely to default on the loan. Proceed accordingly.")

if __name__ == '__main__':
    main()

