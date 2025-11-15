# heart_disease_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
df = pd.read_csv('heart.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nTop 5 Most Important Features:")
print(feature_importance.head())
print("\nVisualizations saved: confusion_matrix.png, feature_importance.png")

# Save results to a text file
with open('model_results.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("HEART DISEASE PREDICTION - MODEL RESULTS\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"Dataset Shape: {df.shape}\n\n")
    
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n")
    
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    
    f.write("Top 5 Most Important Features:\n")
    f.write(feature_importance.head().to_string())
    f.write("\n\n")
    
    f.write("All results and visualizations saved!\n")

print("\nAll results exported to: model_results.txt")