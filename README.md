# LoanPredict — Intelligent Credit Risk Assessment

A full-stack machine learning application for evaluating loan default probability with precision and confidence. Built with a modern fintech aesthetic and powered by five ML models.

**Live Demo:** [loan-default-prediction-neon.vercel.app](https://loan-default-prediction-neon.vercel.app)

---

## Overview

LoanPredict enables financial analysts and institutions to assess the risk of loan default for individual applicants or large batches of customers. Users can input financial and credit profile data and receive an instant default probability score powered by machine learning.

The application features user authentication, prediction history tracking, batch CSV processing, and a full admin dashboard for system monitoring.

---

## Features

- **Individual Assessment** — Enter a customer's financial profile and receive an instant default probability score with identified risk factors
- **Batch Processing** — Upload a CSV file to process hundreds of loan applications simultaneously
- **Five ML Models** — Choose between Random Forest, Logistic Regression, Decision Tree, KNN, and Naive Bayes
- **Prediction History** — All assessments are saved and accessible for audit and review
- **Admin Dashboard** — Monitor system-wide stats, manage users, and track daily activity
- **JWT Authentication** — Secure login and registration with role-based access control

---

## Tech Stack

### Frontend

- React 18 with Vite
- Tailwind CSS
- React Router DOM
- Axios

### Backend

- Node.js + Express
- MongoDB + Mongoose
- JSON Web Tokens (JWT)
- Multer (file uploads)

### ML API

- Python + FastAPI
- Scikit-learn
- Pandas + NumPy
- Joblib

### Deployment

- **Frontend** — Vercel
- **Backend + ML API** — Render
- **Database** — MongoDB Atlas

---

## Architecture

```
React (Vercel)
    │
    ▼
Node.js / Express (Render)
    │           │
    ▼           ▼
MongoDB      FastAPI (Render)
(Atlas)      └── ML Models (.pkl)
```

The React frontend communicates with the Node.js backend, which handles authentication and database operations. For predictions, Node.js forwards requests to the FastAPI ML service, which loads the trained models and returns probability scores.

---

## ML Models

All five models are trained on a real loan dataset with 16 features including credit score, income, debt, bankruptcies, tax liens, and loan details.

| Model               | Description                                   |
| ------------------- | --------------------------------------------- |
| Random Forest       | Ensemble of decision trees — highest accuracy |
| Logistic Regression | Linear model for binary classification        |
| Decision Tree       | Rule-based classification with depth limiting |
| K-Nearest Neighbors | Instance-based learning                       |
| Naive Bayes         | Probabilistic Gaussian classifier             |

SMOTE (Synthetic Minority Oversampling) is applied during training to handle class imbalance.

---

## Project Structure

```
Loan-Default-Prediction/
├── ml_api/                  # Python FastAPI — serves ML models
│   ├── main.py
│   ├── train_models.py
│   ├── utils.py
│   ├── ML_MODEL/            # Trained .pkl files
│   └── Data_sets/
│
├── server/                  # Node.js Express backend
│   ├── index.js
│   ├── routes/
│   │   ├── auth.js
│   │   ├── predict.js
│   │   └── admin.js
│   ├── models/
│   │   ├── User.js
│   │   └── Prediction.js
│   └── middleware/
│       └── auth.js
│
└── client/                  # React frontend
    └── src/
        ├── pages/
        │   ├── Login.jsx
        │   ├── Register.jsx
        │   ├── Dashboard.jsx
        │   ├── Predict.jsx
        │   ├── History.jsx
        │   └── Admin.jsx
        ├── components/
        │   └── Navbar.jsx
        ├── context/
        │   └── AuthContext.jsx
        └── api/
            └── axios.js
```

---

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 20+
- MongoDB (local or Atlas)

### 1. Clone the repository

```bash
git clone https://github.com/SanjeetaAcharya/Loan-Default-Prediction-.git
cd Loan-Default-Prediction-
```

### 2. Train the ML models

```bash
cd ml_api
pip install -r requirements.txt
python train_models.py
```

### 3. Start the FastAPI ML server

```bash
python -m uvicorn main:app --reload --port 8000
```

### 4. Set up and start the Node.js server

```bash
cd ../server
npm install
```

Create a `.env` file inside `server/`:

```
PORT=5000
MONGO_URI=mongodb://localhost:27017/loan_default
JWT_SECRET=your_secret_key
ML_API_URL=http://localhost:8000
```

```bash
npm run dev
```

### 5. Start the React frontend

```bash
cd ../client
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## API Endpoints

### Auth

| Method | Endpoint             | Description           |
| ------ | -------------------- | --------------------- |
| POST   | `/api/auth/register` | Register a new user   |
| POST   | `/api/auth/login`    | Login and receive JWT |
| GET    | `/api/auth/me`       | Get current user      |

### Predictions

| Method | Endpoint               | Description                   |
| ------ | ---------------------- | ----------------------------- |
| POST   | `/api/predict/single`  | Single prediction             |
| POST   | `/api/predict/batch`   | Batch CSV prediction          |
| GET    | `/api/predict/history` | User prediction history       |
| GET    | `/api/predict/models`  | Available models + accuracies |

### Admin

| Method | Endpoint                 | Description       |
| ------ | ------------------------ | ----------------- |
| GET    | `/api/admin/stats`       | System statistics |
| GET    | `/api/admin/users`       | All users         |
| DELETE | `/api/admin/users/:id`   | Delete a user     |
| GET    | `/api/admin/predictions` | All predictions   |

---

## Dataset

The dataset contains loan application records with the following features:

- Credit Score, Annual Income, Monthly Debt
- Tax Liens, Bankruptcies, Number of Credit Problems
- Years in Current Job, Years of Credit History
- Number of Open Accounts, Maximum Open Credit
- Current Loan Amount, Current Credit Balance
- Months Since Last Delinquent
- Home Ownership, Loan Purpose, Loan Term

---

## Deployment

| Service  | Platform      | URL                                                                                        |
| -------- | ------------- | ------------------------------------------------------------------------------------------ |
| Frontend | Vercel        | [loan-default-prediction-neon.vercel.app](https://loan-default-prediction-neon.vercel.app) |
| Backend  | Render        | https://loanpredict-server.onrender.com                                                    |
| ML API   | Render        | https://loanpredict-ml-api.onrender.com                                                    |
| Database | MongoDB Atlas | —                                                                                          |

> **Note:** The backend and ML API are hosted on Render's free tier. They may take 30–50 seconds to wake up after a period of inactivity.

---

## Author

**Sanjeeta Acharya**
[GitHub](https://github.com/SanjeetaAcharya)
