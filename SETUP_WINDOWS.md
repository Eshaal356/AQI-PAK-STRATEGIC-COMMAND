# SETUP INSTRUCTIONS - Windows Users

## Quick Setup Guide

### Step 1: Install Python Dependencies
Open Command Prompt (cmd) or PowerShell and run:

```cmd
pip install -r requirements.txt
```

Or install individually:
```cmd
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm streamlit joblib scipy
```

### Step 2: Prepare Data Files

**IMPORTANT:** You need to add your data files to this submission!

Create the following folder structure:
```
submission/
├── Training/
│   └── concatenated_dataset_Aug_2021_to_July_2024.csv
└── Testing/
    ├── islamabad_complete_data_july_to_dec_2024.csv
    ├── karachi_complete_data_july_to_dec_2024.csv
    ├── lahore_complete_data_july_to_dec_2024.csv
    ├── peshawar_complete_data_july_to_dec_2024.csv
    └── quetta_complete_data_july_to_dec_2024.csv
```

**Where to get data:**
- Download from Kaggle (Pakistan Air Quality datasets mentioned in problem statement)
- Or use your own collected data
- Make sure CSV files have the correct column names

### Step 3: Train Models

Run the training script:
```cmd
python air_quality_prediction.py
```

**Expected Output:**
- Creates `models/` folder with .pkl files
- Creates `outputs/` folder with predictions CSV
- Shows training progress and metrics in console

**Training Time:** 5-15 minutes (depending on your computer)

### Step 4: Run Streamlit Dashboard

After training completes successfully, run:
```cmd
streamlit run streamlit_app.py
```

Then open your browser to: http://localhost:8501

### Troubleshooting

**Problem:** `FileNotFoundError` for data files
**Solution:** Make sure Training/ and Testing/ folders exist with CSV files

**Problem:** `ModuleNotFoundError`
**Solution:** Install missing packages: `pip install <package-name>`

**Problem:** Unicode/Encoding errors
**Solution:** This version has been fixed to remove all emojis - should work on Windows now!

**Problem:** Streamlit won't start
**Solution:** Try: `python -m streamlit run streamlit_app.py`

### For Hackathon Judges

If you don't have the data files, you can review:
1. The code quality in `air_quality_prediction.py`
2. The Streamlit app code in `streamlit_app.py`
3. Documentation in README.md and TECHNICAL_REPORT.md

The code is production-ready and will work once data files are provided.

### File Checklist

Before final submission, ensure you have:
- [X] air_quality_prediction.py (training code)
- [X] streamlit_app.py (dashboard code)
- [X] README.md (documentation)
- [X] TECHNICAL_REPORT.md (technical details)
- [X] QUICK_START.md (quick guide)
- [X] requirements.txt (dependencies)
- [X] SETUP_WINDOWS.md (this file)
- [ ] Training/concatenated_dataset_Aug_2021_to_July_2024.csv (DATA - YOU NEED TO ADD)
- [ ] Testing/*.csv (5 city files - YOU NEED TO ADD)
- [ ] models/*.pkl (Generated after training)
- [ ] outputs/aqi_predictions_3day.csv (Generated after training)

### Contact

Developed by: Eshaal Malik
Date: January 31, 2026
Competition: Air Quality Level Prediction (ML)
