# ❤️ Heart Disease Prediction API with FastAPI & Docker

[![Python](https://img.shields.io/badge/Python-3.12.3-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) 
[![Docker](https://img.shields.io/badge/Docker-24.1.0-blue?logo=docker&logoColor=white)](https://www.docker.com/) 
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Ever wondered if a small API could predict heart health risks in seconds? 🫀 Welcome to **Heart Disease Prediction**, where machine learning meets lightning-fast APIs, all wrapped in Docker magic! This project predicts the likelihood of heart disease using a trained machine learning model, simply feed it health stats like age, cholesterol, blood pressure, and more, and get instant insights. Built with **Python 3.12.3**, **FastAPI**, **Uvicorn**, and **Scikit-learn**, this application provides real-time predictions through a lightweight API, fully containerized with Docker for easy deployment anywhere.  

## 🔍 Project Overview

The application uses a trained ML model to predict heart disease based on user input such as age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate achieved, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia. FastAPI serves as the backend, ensuring lightning-fast responses, while Docker allows consistent deployment across environments.  

# ❤️ Heart Disease Prediction API

**Dataset collected from:** [Heart Disease Dataset — Kaggle (johnsmith88)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

A Machine Learning + FastAPI + Docker project that trains a classifier to predict the presence of heart disease (binary: 0 = no disease, 1 = disease), then serves predictions via a REST API. This repository demonstrates an end-to-end workflow: dataset → model training → model saving (joblib) → FastAPI service → Dockerized deployment.

---

## 🔎 Dataset (collected)
- **Source:** Kaggle — [johnsmith88 / heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
- **File:** `heart.csv`  
- **Target column:** `target` (0 = no heart disease, 1 = heart disease)  
- **Features used:**  
  `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

> **Note:** Please download the dataset from Kaggle (requires a Kaggle account) and place the `heart.csv` into the `model/` folder (or update the path in `model/model_run.py`).

---

## 📥 How to get the dataset
1. Go to the Kaggle dataset page:  
   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. Sign in with your Kaggle account and click **Download**.
3. Extract the downloaded zip and copy `heart.csv` into this repository's `model/` directory:


## ⚡ Features

- 💡 **Instant Predictions**: Input health data, get a prediction.  
- 🐍 **Python 3.12.3 + FastAPI**: Clean, efficient, high-performance API.  
- 🐳 **Dockerized**: Deploy anywhere with Docker containers.  
- 🎯 **ML Model Integration**: Use a trained model for heart disease prediction.  

## 🛠 Tech Stack

- **FastAPI** – Fast Python web framework  
- **Docker** – Containerization for easy deployment  
- **Scikit-learn** – Machine learning model  
- **Uvicorn** – ASGI server for FastAPI  
- **Pydantic** – Data validation  

## 🚀 Installation & Setup

### Prerequisites

- Python 3.12.3  
- Docker  
- Git  

### Clone the Repository

```bash
git clone https://github.com/ZobayerAkib/Heart-Disease-Prediction-FastAPI-Docker-Deployment.git
cd Heart-Disease-Prediction-FastAPI-Docker-Deployment 

