Machine Learning Based Patient Classifier in the Emergency Department

This project is a machine learning-based triage support system designed to classify incoming patients in emergency departments (ED) according to the Korean Triage and Acuity Scale (KTAS). It predicts the appropriate KTAS level based on clinical presentation, vital signs, and other patient details.

Table of Contents:

- Project Overview
- Features
- Architecture
- Getting Started
- Usage
- Model Details
- Dataset
- Team Members
- License

Project Overview:
Emergency departments often experience patient overcrowding and must quickly identify the severity of a patient's condition. The ED Patient Classifier assists medical staff by predicting the KTAS level using supervised machine learning models trained on real emergency visit data.

Features:

- Predicts KTAS levels (1 to 5) for patients
- Built-in data preprocessing and feature engineering
- Multiple ML models: XGBoost, LightGBM, RandomForest, SVM, Logistic Regression
- Interactive Streamlit UI with dark/light theme toggle
- Handles textual data using sentence embeddings
- Includes visualizations such as class distribution before/after SMOTE

Architecture
The system follows a modular architecture with the following components:

- src/ � Source code for preprocessing, model training, and evaluation
- outputs/ � Trained models, plots, and processed data
- app.py � Streamlit-based front-end for user interaction
- evaluate_all_models.py � Evaluation script comparing model performance
- plot_ktas_distribution.py � Script to visualize label distribution before and after SMOTE

Getting Started:

1. Clone the repository
   git clone https://github.com/Charistina/Patient-Classifier-In-The-Emergency-Department-
   cd Patient-Classifier-In-The-Emergency-Department-

2. Create a virtual environment
   python -m venv venv
   source venv/bin/activate
   On Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Download the dataset

- Visit the Emergency Service Triage Application dataset on Kaggle
- Download the dataset CSV file
- Place it in the data/ directory as:
  data/data.csv

5. Run preprocessing and training (if needed)
   If you need to retrain models or reprocess data:
   python -m src.preprocess_data
   python -m src.train_model

6. Run the application
   streamlit run app.py

Usage
Once the app is running:

1. Select a model from the sidebar.
2. Fill out patient information in the input form.
3. Click Predict Triage Level.
4. View the predicted KTAS level along with an interpretation.
   The app supports both light and dark themes and includes explanations for each KTAS level.

Model Details:
The following models were evaluated for performance:

- XGBoost
- LightGBM
- Random Forest
- Logistic Regression
- Support Vector Machine

Model performance was compared using accuracy, F1-score, Cohen's Kappa, and calibration curves. Results are available in the outputs/ directory and generated via evaluate_all_models.py.
Dataset

This project uses the Emergency Service Triage Application dataset from Kaggle.

- Dataset: Emergency Service Triage Application
- Author: Ilker Yildiz
- License: Refer to Kaggle page for licensing details
  Note: The dataset is not included in this repository. You must manually download and place it in the data/ directory as data.csv.

Team Members:
This project was completed as a team effort. Team members are listed below:

- Christina Charis R
- Annaparthi Sandhya
- A. Silvia Jasmine

License:
This project is for academic purposes only. Please refer to the dataset�s original license for usage and redistribution rights.
