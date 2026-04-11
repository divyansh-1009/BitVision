# BitVision: Bitcoin Forecasting Model

Live deployment: https://bitvision-iitj.streamlit.app/

## Running the App and Notebooks

Inside the code folder,

### 1) Create virtual environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the notebook by selecting kernel as venv or run the app as

```bash
streamlit run app/Home.py
```

### 4) Open in browser

Streamlit will print a local URL (usually `http://localhost:8501`).

## Project Structure

```text
code/
├── app/                # Streamlit dashboard source code
│   ├── Home.py         # App entry point
│   ├── pages/          # Multi-page dashboard views
│   └── utils/          # Helper functions for data & inference
├── data/               # Training and comparison datasets (CSV)
├── models/             # Trained model artifacts (.pkl)
├── notebooks/          # Research, EDA, and model experiments
│   └── experiments/    # Detailed ML/DL comparison notebooks
├── requirements.txt    # Python dependencies
```
## Problem Statement

Bitcoin prices are highly volatile and difficult to forecast directly from raw OHLCV data.  
This project addresses the problem by predicting the closing price of Bitcoin of the next day using OHLCV and other technical indicators. 

BitVision is an end-to-end Bitcoin forecasting project that combines time-series feature engineering, model experimentation, and an interactive Streamlit dashboard for analysis and predictions.

## Team Members 

- Divyansh Yadav
- Akhil Dhyani
- Harshit
- Gaurang Goyal


