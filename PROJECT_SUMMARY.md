# 📖 CS2 Time Series Analytics: Project Narrative & Deep Dive

This document provides a comprehensive walkthrough of the project's lifecycle, from initial data concepts to the final production-grade interactive dashboard. Use this as a guide for explaining the project in interviews or portfolio presentations.

---

## 🎯 1. Project Vision
The objective was to build a professional-grade time series analysis system that evaluates the **Counter-Strike 2 (CS2) ecosystem**. Unlike basic forecasting projects, this system integrates three distinct data streams—**Player Counts**, **Search Interest (Google Trends)**, and **Viewership (Twitch)**—to find leading indicators of player engagement.

---

## 📊 2. Phase 1: Data Engineering & Synthetic Generation
The foundation of the project is a multivariate dataset covering the transition from CS:GO to CS2.
- **Source**: Integrated Steam player data with search and stream metrics.
- **Challenge**: Handling the structural break caused by the CS2 release in late 2023.
- **Data Integrity**: Handled historical limitations, such as Twitch data only being available from late 2016 onwards, ensuring the models ignore missing values during calculation without breaking the pipeline.

---

## 📓 3. Phase 2: Exploratory Data Analysis (The Notebook)
Before building the dashboard, the project began in `TSA_mini.ipynb`. This was the "research sandbox" where:
- Raw data was cleaned and visualized.
- The **ADF (Augmented Dickey-Fuller)** test was used to prove that raw player counts are non-stationary.
- Initial **ARIMA** and **VAR** models were prototyped to see if search interest actually "Granger-caused" player growth.
- This phase proved the concept: **Google Trends is a 3-5 month leading indicator** for the game's health.

---

## 🏗️ 4. Phase 3: Productionalizing the Codebase
To move beyond a simple script, I re-engineered the logic into a modular, production-ready architecture:
- **`data_loader.py`**: Isolated data ingestion with Streamlit caching (`@st.cache_data`) for high performance.
- **`models.py`**: A pure mathematical engine. By removing all UI logic from the models, I ensured the code is **unit-testable**.
- **`config.yaml`**: Externalized all model hyperparameters, allowing for quick tuning without touching the source code.

---

## 🧪 5. Phase 4: Advanced Statistical Methodology
The project utilizes a "Full Stack" of econometric models:
- **Stationarity Management**: Taking the "First Difference" of non-stationary series to ensure model validity.
- **Univariate Forecasting (ARIMA)**: Predicting player counts based on their own historical momentum.
- **Multivariate Forecasting (VAR)**: Predicting players based on the *combined* system of search and stream data.
- **Causality Testing (Granger)**: Mathematically proving that one variable "leads" another.
- **Volatility Modeling (GARCH)**: Borrowing from finance to measure how long "shocks" (like a new game launch) stay chaotic in the system.

---

## 🎮 6. Phase 5: The "Esports Dashboard" (UI/UX)
The final product is a **Counter-Strike 2 Tactical Dashboard** built in Streamlit.
- **Theme**: A custom-coded "Gunmetal" dark mode with official **CT Blue** and **T Gold** accents.
- **Interactivity**: Users can dynamically slide the forecast horizon (3-24 months) and adjust confidence intervals to see how risk grows over time.
- **Automated Insights**: Injected "Smart Analysis" boxes that explain complex statistical results in plain English for non-technical stakeholders.

---

## 🛡️ 7. Engineering Excellence (CI/CD)
To ensure the project meets professional software standards, I implemented:
- **Unit Testing**: A robust suite of `pytest` tests covering all mathematical functions.
- **CI/CD Pipeline**: A GitHub Actions workflow that automatically runs **Flake8** (linting) and **Black** (formatting) on every push to ensure code quality.
- **Type Safety & PEP 8**: Strictly adhered to Python best practices for readability and maintenance.

---

## 🏁 Summary for Interviews
*"In this project, I proved that search volume leads player counts by 3-5 months using Granger Causality. I built a modular Python system that transitions from raw CSV data to a professional, interactive dashboard, ensuring all models are statistically validated through ADF and GARCH tests while maintaining high code quality through CI/CD pipelines."*
