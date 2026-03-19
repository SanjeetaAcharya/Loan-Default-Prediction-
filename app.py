def main():
    st.title('Loan Default Prediction')
    selected_model = st.selectbox("Choose ML Model", list(model_options.keys()))
    pipeline = joblib.load(model_options[selected_model])

    st.write('Enter customer information to predict loan default.')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Customer Information')

        credit_score = st.slider('Credit Score', 300, 850, 700,
            help="Below 650 is considered high risk")
        annual_income = st.number_input('Annual Income ($)', min_value=0, value=50000,
            help="Your yearly income")
        monthly_debt = st.number_input('Monthly Debt ($)', min_value=0, value=1000,
            help="Total monthly debt payments")
        tax_liens = st.slider('Tax Liens', 0, 10, 0,
            help="Number of tax liens — any value above 0 increases risk")
        bankruptcies = st.slider('Bankruptcies', 0, 5, 0,
            help="Number of bankruptcies — any value above 0 increases risk")
        credit_problems = st.slider('Number of Credit Problems', 0, 10, 0,
            help="More than 1 increases default risk significantly")
        home_ownership = st.selectbox('Home Ownership',
            ['Have Mortgage', 'Home Mortgage', 'Own Home', 'Rent'])
        purpose = st.selectbox('Loan Purpose', [
            'business loan', 'buy a car', 'buy house', 'debt consolidation',
            'educational expenses', 'home improvements', 'major purchase',
            'medical bills', 'moving', 'other', 'small business',
            'take a trip', 'vacation', 'wedding'
        ])
        term = st.selectbox('Loan Term', ['Short Term', 'Long Term'])

    input_data = pd.DataFrame({
        'Credit Score': [credit_score],
        'Annual Income': [annual_income],
        'Monthly Debt': [monthly_debt],
        'Tax Liens': [tax_liens],
        'Bankruptcies': [bankruptcies],
        'Number of Credit Problems': [credit_problems],
        'Home Ownership': [home_ownership],
        'Purpose': [purpose],
        'Term': [term]
    })

    categorical_columns = ['Home Ownership', 'Purpose', 'Term']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns)
    feature_columns = joblib.load('ML_MODEL/feature_columns.pkl')
    input_data_encoded = input_data_encoded.reindex(columns=feature_columns, fill_value=0)
    input_data_scaled = pipeline.named_steps['scaler'].transform(input_data_encoded)

    classifier = pipeline.named_steps['classifier']
    prediction = classifier.predict(input_data_scaled)
    probability = classifier.predict_proba(input_data_scaled)[0][1]

    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=['Will Not Default', 'Will Default'],
                y=[1 - probability, probability],
                palette=['green', 'red'],
                ax=ax
            )
            ax.set_title('Loan Default Probability')
            ax.set_ylabel('Probability')
            st.pyplot(fig)

            if prediction[0] == 1:
                st.error("⚠️ HIGH RISK — This customer is likely to default.")
                st.metric(label="Default Probability", value=f"{probability:.1%}",
                    delta="High Risk", delta_color="inverse")
            else:
                st.success("✅ LOW RISK — This customer is not likely to default.")
                st.metric(label="Default Probability", value=f"{probability:.1%}",
                    delta="Low Risk", delta_color="normal")

            # Show what's driving the risk
            st.subheader("🔍 Risk Factors")
            if credit_score < 650:
                st.warning("⚠️ Credit score below 650")
            if tax_liens > 0:
                st.warning(f"⚠️ {tax_liens} tax lien(s) found")
            if bankruptcies > 0:
                st.warning(f"⚠️ {bankruptcies} bankruptcy/bankruptcies found")
            if credit_problems > 1:
                st.warning(f"⚠️ {credit_problems} credit problems found")
            if annual_income > 0 and (monthly_debt / (annual_income / 12)) > 0.5:
                st.warning("⚠️ Monthly debt exceeds 50% of monthly income")

if __name__ == '__main__':
    main()