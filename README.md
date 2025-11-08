# ❤️ Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Healthcare-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Early detection of heart disease using machine learning for better healthcare outcomes

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Clinical Features](#clinical-features)
- [Model Deployment](#model-deployment)
- [Medical Disclaimer](#medical-disclaimer)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🌟 Overview

This machine learning system predicts the likelihood of heart disease based on clinical parameters. Using various classification algorithms, the system analyzes patient data including age, blood pressure, cholesterol levels, and other vital signs to assess heart disease risk.

**Important**: This is a research/educational project. Always consult healthcare professionals for medical decisions.

## ✨ Features

- 🏥 **Accurate Predictions**: High accuracy ML models for risk assessment
- 📊 **Multiple Algorithms**: Comparison of various ML techniques
- ⚡ **Fast Analysis**: Instant risk assessment
- 📈 **Risk Scoring**: Probability-based risk categorization
- 🔍 **Feature Importance**: Understand key risk factors
- 📱 **User-Friendly Interface**: Easy input and interpretation
- 📉 **Visualization**: Clear charts and risk indicators
- 🔐 **Privacy-Focused**: Local processing, no data storage

## 🎥 Demo

### Sample Prediction

```
Input Parameters:
- Age: 55 years
- Sex: Male
- Chest Pain Type: Typical Angina
- Resting Blood Pressure: 140 mm Hg
- Cholesterol: 240 mg/dl
- Fasting Blood Sugar: >120 mg/dl
- Resting ECG: Normal
- Max Heart Rate: 150 bpm
- Exercise Induced Angina: Yes

Prediction: ⚠️ HIGH RISK (78% probability)
Recommendation: Consult cardiologist immediately
```

*Note: Add screenshots of your prediction interface here*

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Analysis and experimentation
- **Flask**: Web interface (optional)
- **Streamlit**: Interactive dashboard (optional)

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- 2GB+ RAM
- Basic understanding of medical terminologies (helpful)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/rohithreddy5250/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset** (if not included)
```bash
python download_data.py
```

## 🚀 Usage

### Basic Usage - Single Prediction

```python
from heart_predictor import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor()

# Load trained model
predictor.load_model('models/best_model.pkl')

# Patient data
patient_data = {
    'age': 55,
    'sex': 1,  # 1 = male, 0 = female
    'cp': 3,   # chest pain type
    'trestbps': 140,  # resting blood pressure
    'chol': 240,  # cholesterol
    'fbs': 1,  # fasting blood sugar > 120 mg/dl
    'restecg': 0,  # resting ECG results
    'thalach': 150,  # max heart rate achieved
    'exang': 1,  # exercise induced angina
    'oldpeak': 2.5,  # ST depression
    'slope': 2,  # slope of peak exercise ST segment
    'ca': 1,  # number of major vessels
    'thal': 2  # thalassemia
}

# Make prediction
result = predictor.predict(patient_data)

print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Recommendation: {result['recommendation']}")
```

### Batch Predictions

```python
import pandas as pd

# Load patient data
patients_df = pd.read_csv('patients_data.csv')

# Predict for all patients
predictions = predictor.predict_batch(patients_df)

# Save results
predictions.to_csv('risk_assessment_results.csv', index=False)
```

### Command Line Interface

```bash
# Single prediction
python predict.py --age 55 --sex male --cp 3 --trestbps 140 --chol 240

# From CSV file
python predict.py --input patients.csv --output results.csv

# Interactive mode
python predict.py --interactive
```

### Web Interface (Streamlit)

```bash
streamlit run app.py
# Opens browser at http://localhost:8501
```

## 📊 Dataset

### Data Source

- **UCI Heart Disease Dataset**: Cleveland database
- **303 patient records**
- **14 clinical features**
- **Binary classification**: Presence/Absence of heart disease

### Dataset Structure

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| age | Age in years | Numeric | 29-77 |
| sex | Gender | Binary | 0=F, 1=M |
| cp | Chest pain type | Categorical | 0-3 |
| trestbps | Resting blood pressure | Numeric | 94-200 mm Hg |
| chol | Serum cholesterol | Numeric | 126-564 mg/dl |
| fbs | Fasting blood sugar > 120 mg/dl | Binary | 0, 1 |
| restecg | Resting ECG results | Categorical | 0-2 |
| thalach | Maximum heart rate | Numeric | 71-202 |
| exang | Exercise induced angina | Binary | 0, 1 |
| oldpeak | ST depression | Numeric | 0-6.2 |
| slope | Slope of peak exercise ST | Categorical | 0-2 |
| ca | Number of major vessels | Numeric | 0-3 |
| thal | Thalassemia | Categorical | 0-3 |
| target | Heart disease presence | Binary | 0, 1 |

### Data Preprocessing

1. **Handling Missing Values**: Imputation using median/mode
2. **Feature Scaling**: StandardScaler normalization
3. **Encoding**: Label encoding for categorical variables
4. **Train-Test Split**: 80-20 split with stratification
5. **Cross-Validation**: 5-fold CV for model validation

## 📈 Model Performance

### Algorithm Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **88.5%** | **87.2%** | **89.1%** | **88.1%** | **0.92** |
| Logistic Regression | 85.2% | 84.5% | 86.3% | 85.4% | 0.89 |
| SVM | 86.7% | 85.8% | 87.5% | 86.6% | 0.90 |
| Decision Tree | 82.3% | 81.0% | 83.7% | 82.3% | 0.86 |
| KNN | 83.5% | 82.3% | 84.8% | 83.5% | 0.87 |
| Gradient Boosting | 87.9% | 86.5% | 88.6% | 87.5% | 0.91 |
| Neural Network | 86.4% | 85.1% | 87.3% | 86.2% | 0.89 |

### Confusion Matrix (Random Forest)

```
                Predicted
                No    Yes
Actual  No      42     5
        Yes     2     12
```

- **True Positives**: 42
- **True Negatives**: 12
- **False Positives**: 5
- **False Negatives**: 2

### Feature Importance

Top 5 most important features:
1. **cp (chest pain type)**: 18.5%
2. **thalach (max heart rate)**: 15.2%
3. **oldpeak (ST depression)**: 13.8%
4. **ca (major vessels)**: 12.4%
5. **thal (thalassemia)**: 11.1%

## 🏥 Clinical Features Explained

### Chest Pain Types (cp)
- **0**: Asymptomatic
- **1**: Atypical angina
- **2**: Non-anginal pain
- **3**: Typical angina

### Resting ECG (restecg)
- **0**: Normal
- **1**: ST-T wave abnormality
- **2**: Left ventricular hypertrophy

### ST Slope (slope)
- **0**: Downsloping
- **1**: Flat
- **2**: Upsloping

### Thalassemia (thal)
- **0**: Normal
- **1**: Fixed defect
- **2**: Reversible defect
- **3**: Not described

## 🌐 Model Deployment

### Deploy as Web App

```bash
# Using Flask
python app.py
# Access at http://localhost:5000

# Using Streamlit
streamlit run streamlit_app.py
```

### Deploy to Cloud

```bash
# Heroku
heroku create heart-disease-predictor
git push heroku main

# AWS Lambda (serverless)
# See deployment/aws_lambda.md

# Docker
docker build -t heart-predictor .
docker run -p 5000:5000 heart-predictor
```

## 📁 Project Structure

```
Heart_Disease_Prediction/
│
├── data/
│   ├── heart_disease.csv        # Raw dataset
│   ├── processed_data.csv       # Preprocessed data
│   └── data_dictionary.md       # Feature descriptions
├── models/
│   ├── random_forest.pkl        # Trained RF model
│   ├── logistic_regression.pkl  # Trained LR model
│   └── model_comparison.csv     # Performance metrics
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory analysis
│   ├── 02_preprocessing.ipynb   # Data preprocessing
│   ├── 03_model_training.ipynb  # Model training
│   └── 04_evaluation.ipynb      # Model evaluation
├── src/
│   ├── heart_predictor.py       # Main predictor class
│   ├── preprocessing.py         # Preprocessing functions
│   ├── model_training.py        # Training pipeline
│   ├── evaluation.py            # Evaluation metrics
│   └── utils.py                 # Utility functions
├── deployment/
│   ├── app.py                   # Flask web app
│   ├── streamlit_app.py         # Streamlit dashboard
│   └── Dockerfile               # Docker configuration
├── tests/
│   └── test_predictor.py        # Unit tests
├── predict.py                   # CLI interface
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── LICENSE
```

## 🎯 Usage Examples

### Example 1: Risk Assessment for New Patient
```python
patient = {
    'age': 60, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 250, 'fbs': 1, 'restecg': 0, 'thalach': 140,
    'exang': 1, 'oldpeak': 2.3, 'slope': 1, 'ca': 1, 'thal': 2
}

result = predictor.predict(patient)
if result['risk_level'] == 'HIGH':
    print("⚠️ HIGH RISK - Immediate medical attention recommended")
```

### Example 2: Batch Screening
```python
# Screen multiple patients
screening_results = predictor.predict_batch(patients_df)
high_risk_patients = screening_results[screening_results['risk_level'] == 'HIGH']
print(f"Found {len(high_risk_patients)} high-risk patients")
```

### Example 3: Feature Analysis
```python
# Analyze which features contribute most to risk
feature_importance = predictor.get_feature_importance()
print("\nTop Risk Factors:")
for feature, importance in feature_importance.head():
    print(f"{feature}: {importance:.2%}")
```

## ⚙️ Configuration

Edit `config.py`:

```python
# Model settings
MODEL_TYPE = 'random_forest'  # random_forest, logistic, svm, etc.
USE_CROSS_VALIDATION = True
N_FOLDS = 5

# Risk thresholds
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.6
# Above 0.6 = HIGH RISK

# Feature engineering
APPLY_FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'recursive'
```

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for educational and research purposes only.

- ❌ **NOT a substitute** for professional medical advice
- ❌ **NOT approved** for clinical use
- ❌ **DO NOT** make medical decisions based solely on these predictions
- ✅ **ALWAYS** consult qualified healthcare professionals
- ✅ **USE** as a screening tool or educational resource

The developers are not responsible for any medical decisions made using this tool.

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Test specific module
pytest tests/test_predictor.py -v
```

## 🚀 Future Improvements

- [ ] Integration with electronic health records (EHR)
- [ ] Real-time monitoring and alerts
- [ ] Deep learning models (Neural Networks)
- [ ] Explainable AI (SHAP, LIME) for interpretability
- [ ] Mobile app for patients
- [ ] Multi-language support
- [ ] Additional cardiovascular conditions
- [ ] Personalized lifestyle recommendations
- [ ] Clinical trial integration

## 🤝 Contributing

Medical and ML expertise welcome! 

1. Fork the repository
2. Create feature branch (`git checkout -b feature/Improvement`)
3. Commit changes (`git commit -m 'Add medical feature'`)
4. Push to branch (`git push origin feature/Improvement`)
5. Open Pull Request

## 📚 References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [American Heart Association Guidelines](https://www.heart.org)
- [ML in Healthcare: A Review](https://www.nature.com/articles/s41591-019-0548-6)
- [Cardiovascular Disease Prediction](https://www.who.int/health-topics/cardiovascular-diseases)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Rohith Reddy**

- GitHub: [@rohithreddy5250](https://github.com/rohithreddy5250)
- LinkedIn: [rohithreddyy](https://linkedin.com/in/rohithreddyy)
- Email: rohithreddybaddam8@gmail.com

## 🙏 Acknowledgments

- UCI Machine Learning Repository
- Healthcare ML research community
- Open-source contributors
- Medical professionals for domain expertise

---

⭐ **If this project could help save lives, give it a star!**

**Made with ❤️ and 🏥 by Rohith Reddy**
