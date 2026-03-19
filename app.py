import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_years_in_job

model_options = {
    "KNN": "ML_MODEL/knn_model.pkl",
    "Decision Tree": "ML_MODEL/decision_tree_model.pkl",
    "Logistic Regression": "ML_MODEL/logistic_regression_model.pkl",
    "Naive Bayes": "ML_MODEL/naive_bayes_model.pkl",
    "Random Forest": "ML_MODEL/random_forest_model.pkl"
}

num_features = [
    'Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts',
    'Years of Credit History', 'Maximum Open Credit', 'Number of Credit Problems',
    'Months since last delinquent', 'Bankruptcies', 'Current Loan Amount',
    'Current Credit Balance', 'Monthly Debt', 'Credit Score'
]
categorical_columns = ['Home Ownership', 'Purpose', 'Term']

def main():
    st.title('Loan Default Prediction')
    selected_model = st.selectbox("Choose ML Model", list(model_options.keys()))
    pipeline = joblib.load(model_options[selected_model])

    st.write('Enter customer information to predict loan default.')

    if st.button('🎲 Fill with Sample Data'):
        st.session_state['annual_income'] = 500000
        st.session_state['years_in_job'] = 5
        st.session_state['tax_liens'] = 0
        st.session_state['open_accounts'] = 10
        st.session_state['credit_history'] = 15.0
        st.session_state['max_credit'] = 300000
        st.session_state['credit_problems'] = 0
        st.session_state['months_delinquent'] = 0
        st.session_state['bankruptcies'] = 0
        st.session_state['loan_amount'] = 150000
        st.session_state['credit_balance'] = 80000
        st.session_state['monthly_debt'] = 5000
        st.session_state['credit_score'] = 720

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Customer Information')

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

        home_ownership = st.selectbox('Home Ownership', ['Have Mortgage', 'Home Mortgage', 'Own Home', 'Rent'])
        purpose = st.selectbox('Purpose', [
            'business loan', 'buy a car', 'buy house', 'debt consolidation',
            'educational expenses', 'home improvements', 'major purchase',
            'medical bills', 'moving', 'other', 'small business',
            'take a trip', 'vacation', 'wedding'
        ])
        term = st.selectbox('Term', ['Short Term', 'Long Term'])

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

    input_data = preprocess_years_in_job(input_data)
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns)
    feature_names = pipeline.named_steps['scaler'].get_feature_names_out()
    input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)
    input_data_scaled = pipeline.named_steps['scaler'].transform(input_data_encoded)

    classifier = pipeline.named_steps['classifier']
    prediction = classifier.predict(input_data_scaled)
    probability = classifier.predict_proba(input_data_scaled)[0][1]

    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            sns.barplot(x=['Will Not Default', 'Will Default'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
            axes[0].set_title('Loan Default Probability')
            axes[0].set_ylabel('Probability')

            sns.histplot(input_data['Credit Score'], kde=True, ax=axes[1])
            axes[1].set_title('Credit Score Distribution')

            st.pyplot(fig)

            if prediction[0] == 1:
                st.error("⚠️ HIGH RISK — This customer is likely to default on the loan.")
                st.metric(label="Default Probability", value=f"{probability:.1%}", delta="High Risk", delta_color="inverse")
            else:
                st.success("✅ LOW RISK — This customer is not likely to default on the loan.")
                st.metric(label="Default Probability", value=f"{probability:.1%}", delta="Low Risk", delta_color="normal")

if __name__ == '__main__':
    main()
