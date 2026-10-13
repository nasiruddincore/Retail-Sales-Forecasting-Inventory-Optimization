# 📦 AI-Powered Retail Sales Forecasting & Inventory Optimization System

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Frontend](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

An industry-oriented data science platform designed to predict item-level retail sales and automate supply chain replenishment decisions. By combining Machine Learning forecasting with traditional Operations Research (EOQ, ROP, Safety Stock), this system helps retail businesses prevent costly stockouts while minimizing excessive overstocking.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Objectives](#-system-objectives)
- [Industry Relevance & Use Cases](#-industry-relevance--use-cases)
- [Architecture & Data Flow](#-architecture--data-flow)
- [Folder Structure](#-folder-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run (Execution Workflow)](#-how-to-run-execution-workflow)
- [Author & License](#-author--license)

---

## 📌 Project Overview
Retail inventory management requires balancing supply and demand perfectly. This project uses a Random Forest machine learning model to study historical sales data, promotional lifts, and seasonal trends to forecast future demand. The trained model is deployed through an interactive Streamlit web dashboard, enabling supply chain planners to calculate exact restock quantities instantly.

---

## ⚠️ Problem Statement
* **Stockouts:** Missing inventory leads to lost revenue and customer churn.
* **Overstocking:** Excess inventory ties up working capital and increases holding/warehousing costs.
* **Manual Bottlenecks:** Traditional forecasting relies on slow, error-prone spreadsheet calculations. This project automates the workflow from raw data to final Purchase Order (PO) recommendations.

---

## 🎯 System Objectives
1. **Virtual POS Telemetry Generation:** Simulate a realistic, continuous daily sales dataset incorporating weekend boosts, promotional lifts, and stockout censoring.
2. **Time-Series Feature Extraction:** Construct auto-regressive lag observations and rolling averages to capture purchasing behaviors.
3. **Predictive Modeling:** Train a Scikit-Learn Random Forest Regressor to predict future daily unit sales.
4. **Operations Research Integration:** Translate raw ML forecasts into Safety Stock, Reorder Points (ROP), and Economic Order Quantity (EOQ).
5. **Interactive Dashboard:** Provide a Streamlit UI for managers to simulate supplier lead times and service levels in real-time.

---

## 🏭 Industry Relevance & Use Cases
* **Supermarkets & Grocery (D-Mart, Walmart):** Optimizes shelf replenishment for perishable goods with strict shelf lives.
* **E-Commerce (Amazon, Flipkart):** Balances warehouse load and predicts fulfillment center demand across thousands of SKUs.
* **Direct-to-Consumer (D2C) Brands:** Calculates exact factory reorder quantities based on variable overseas shipping lead times.

---

## 📐 Architecture & Data Flow

```text
[ Virtual POS Simulator ] ──(Generates Sales Data)──► [ data/retail_timeseries.csv ]
                                                              │
                                                    (Feature Engineering)
                                                              ▼
[ Streamlit UI Dashboard ] ◄──────(Inference)─────── [ Random Forest ML Model ]
      (Calculates ROP, EOQ, Safety Stock)

📂 Folder Structure
Plaintext

Retail-Sales-Forecasting-Inventory-Optimization/
│
├── assets/                  # Proof-of-work screenshots for GitHub
│   ├── dashboard_full.png
│   └── terminal_training.png
│
├── data/                    # Generated synthetic retail dataset
│   └── retail_timeseries.csv
│
├── models/                  # Serialized trained ML models
│   └── retail_forecast_model.pkl
│
├── src/                     # Core Python logic
│   ├── simulate_data.py     # Virtual retail POS data generator
│   ├── train_model.py       # ML training and evaluation pipeline
│   └── inventory_logic.py   # Supply chain mathematics (EOQ, ROP)
│
├── app/                     # Web visualization frontend
│   └── app.py               # Streamlit interactive UI
│
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

⚙️ Installation & Setup

1. Clone the Repository
Bash

git clone [https://github.com/YOUR_USERNAME/AI-Retail-Inventory-Optimizer.git](https://github.com/YOUR_USERNAME/AI-Retail-Inventory-Optimizer.git)
cd AI-Retail-Inventory-Optimizer

2. Create a Virtual Environment
Bash

python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

3. Install Dependencies
Bash

pip install -r requirements.txt

🚀 How to Run (Execution Workflow)

Step 1: Generate the Virtual Retail Data

Creates realistic daily sales records for multiple stores and SKUs.
Bash

python src/simulate_data.py

Step 2: Train the Machine Learning Model

Extracts time-series features and trains the Random Forest regressor.
Bash

python src/train_model.py

Step 3: Launch the Real-Time Dashboard

Starts the Streamlit web application to visualize forecasts and restock plans.
Bash

streamlit run app/app.py

(Open your browser to http://localhost:8501 to view the interactive dashboard.)
👤 Author & License

Developed as an industry-oriented Supply Chain Analytics & Machine Learning portfolio project. Distributed under the MIT License.
