# Heart Disease Prediction - Supervised Learning Project

## Project Structure
```
heart-disease-prediction/
│
├── src/                           # Source code
│   └── heart_disease_prediction.py  # Main script
│
├── data/                          # Dataset folder
│   └── heart.csv                  # Heart disease dataset
│
├── outputs/                       # Generated results
│   ├── confusion_matrix.png       # Model prediction visualization
│   ├── feature_importance.png     # Feature importance chart
│   └── model_results.txt          # Detailed results and metrics
│
├── README.md                      # Project documentation
```

## Project Overview
This project implements a **supervised learning** classification model to predict the presence of heart disease in patients using clinical data. The model uses a Random Forest Classifier trained on the UCI Heart Disease dataset, which contains 14 medical attributes such as age, cholesterol levels, chest pain type, and resting blood pressure. By analyzing these features, the model learns patterns that distinguish patients with and without heart disease, achieving accurate predictions that could assist in early detection and medical decision-making.

## Dataset
- **Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Size**: 303 samples with 14 features
- **Target Variable**: Binary (0 = no disease, 1 = disease present)

## Features
The dataset includes:
- `age`: Age of the patient
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia type
- `target`: Heart disease presence (0 or 1)

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
1. Download the dataset from Kaggle
2. Place `heart.csv` in the same directory as the script
3. Run the script:
```bash
python heart_disease_prediction.py
```

## Model Details
- **Algorithm**: Random Forest Classifier
- **Train/Test Split**: 80/20
- **Feature Scaling**: StandardScaler
- **Hyperparameters**: 100 estimators, random_state=42

## Output
- Model accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Feature importance plot
- Top contributing features for prediction

## Results
The model provides insights into which clinical factors are most predictive of heart disease, helping to understand the key indicators medical professionals should monitor.

## License
This project uses publicly available data for educational purposes.