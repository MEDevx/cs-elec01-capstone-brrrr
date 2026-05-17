# USM Student Retention Early Warning System

A Streamlit dashboard that predicts student dropout risks using a custom-built Random Forest algorithm (pure Python) analyzing 36 academic and demographic variables.

## File Structure

```
usm-ews/
├── app.py # Main application
├── style.css # Stylesheet
├── data.csv # Dataset
├── requirements.txt # Dependencies
└── assets/
   └── usm_seal.png # USM logo
```

## Setup & Run

1. Ensure all files, including `data.csv` (semicolon-separated), are in the same folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   streamlit run app.py
   ```
