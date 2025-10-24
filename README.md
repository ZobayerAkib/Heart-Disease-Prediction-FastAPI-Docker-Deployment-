# ❤️ Heart Disease Prediction API
*A Machine Learning + FastAPI + Docker project by <b>Md. Zobayer Ibna Kabir</b>*

---

## 📘 Overview
This project demonstrates how to train and deploy a **machine learning model** for predicting the presence of heart disease using **FastAPI**.  
The app is fully containerized with **Docker** and can be launched with a single command using **docker-compose**.

---

## 🧠 Dataset
Dataset: [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

- **Target variable:** `target` → `1` = Heart disease present, `0` = No heart disease  
- **Input features:**
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`,
    `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## 🏗️ Project Structure

Heart-Disease-Prediction-FastAPI-Docker-Deployment-/
│
├── model/
│ ├── heart_model.joblib # Saved trained model
│ ├── scaler.joblib # StandardScaler for preprocessing
│ └── model_run.py # Script to train and save model
│
├── app/
│ ├── main.py # FastAPI app
│ └── schemas.py # Pydantic models for validation
│
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
├── docker-compose.yml # Docker Compose setup
└── README.md # Project documentation

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ZobayerAkib/Heart-Disease-Prediction-FastAPI-Docker-Deployment-.git
cd Heart-Disease-Prediction-FastAPI-Docker-Deployment-


2️⃣ (Optional) Create a Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
