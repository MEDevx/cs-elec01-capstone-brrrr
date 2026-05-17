# USM Student Retention Early Warning System

A Streamlit dashboard that predicts student dropout risks using a custom-built Random Forest algorithm (pure Python) analyzing 36 academic and demographic variables.

## File Structure

```text
usm-ews/
|-- app.py # Streamlit entrypoint
|-- style.css # Stylesheet
|-- data.csv # Dataset
|-- requirements.txt # Dependencies
|-- assets/
|   `-- usm_seal.png # USM logo
`-- usm_ews/
    |-- constants.py # Mappings and labels
    |-- data.py # Data loading and training
    |-- layout.py # Page config and shared layout
    |-- model.py # Custom Random Forest implementation
    |-- prediction.py # Prediction and risk derivation
    |-- sections.py # Main dashboard sections
    `-- sidebar.py # Sidebar input form
```

## Setup & Run

1. Ensure all files, including `data.csv` (semicolon-separated), are in the same folder.
2. Install the required dependencies:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```
   If Windows still tries to build `pandas` from source, use a 64-bit Python 3.12+ install.
3. Launch the application:
   ```bash
   streamlit run app.py
   ```
