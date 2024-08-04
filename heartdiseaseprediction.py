import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

# Load the Heart Disease dataset from a CSV file
file_path = 'G:\C Data\heart.csv'  # Update this path to your CSV file
data = pd.read_csv(file_path)

# Prepare features and target
features = data.drop('target', axis=1)
target = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create the GUI application using tkinter
def predict_heart_disease():
    try:
        age = float(age_entry.get())
        sex = int(sex_entry.get())
        cp = int(cp_entry.get())
        trestbps = float(trestbps_entry.get())
        chol = float(chol_entry.get())
        fbs = int(fbs_entry.get())
        restecg = int(restecg_entry.get())
        thalach = float(thalach_entry.get())
        exang = int(exang_entry.get())
        oldpeak = float(oldpeak_entry.get())
        slope = int(slope_entry.get())
        ca = int(ca_entry.get())
        thal = int(thal_entry.get())

        # Create a feature vector
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Standardize the features
        features = scaler.transform(features)

        # Make prediction
        prediction = svm_classifier.predict(features)

        # Show result
        if prediction[0] == 1:
            result = "The patient is likely to have heart disease."
        else:
            result = "The patient is unlikely to have heart disease."

        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Heart Disease Prediction")

# Create and place the labels and entries
tk.Label(root, text="Age").grid(row=0, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1)

tk.Label(root, text="Sex (1=Male, 0=Female)").grid(row=1, column=0)
sex_entry = tk.Entry(root)
sex_entry.grid(row=1, column=1)

tk.Label(root, text="Chest Pain Type (0-3)").grid(row=2, column=0)
cp_entry = tk.Entry(root)
cp_entry.grid(row=2, column=1)

tk.Label(root, text="Resting Blood Pressure").grid(row=3, column=0)
trestbps_entry = tk.Entry(root)
trestbps_entry.grid(row=3, column=1)

tk.Label(root, text="Serum Cholestoral in mg/dl").grid(row=4, column=0)
chol_entry = tk.Entry(root)
chol_entry.grid(row=4, column=1)

tk.Label(root, text="Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)").grid(row=5, column=0)
fbs_entry = tk.Entry(root)
fbs_entry.grid(row=5, column=1)

tk.Label(root, text="Resting Electrocardiographic Results (0-2)").grid(row=6, column=0)
restecg_entry = tk.Entry(root)
restecg_entry.grid(row=6, column=1)

tk.Label(root, text="Maximum Heart Rate Achieved").grid(row=7, column=0)
thalach_entry = tk.Entry(root)
thalach_entry.grid(row=7, column=1)

tk.Label(root, text="Exercise Induced Angina (1=Yes, 0=No)").grid(row=8, column=0)
exang_entry = tk.Entry(root)
exang_entry.grid(row=8, column=1)

tk.Label(root, text="ST Depression Induced by Exercise").grid(row=9, column=0)
oldpeak_entry = tk.Entry(root)
oldpeak_entry.grid(row=9, column=1)

tk.Label(root, text="Slope of the Peak Exercise ST Segment (0-2)").grid(row=10, column=0)
slope_entry = tk.Entry(root)
slope_entry.grid(row=10, column=1)

tk.Label(root, text="Number of Major Vessels (0-3)").grid(row=11, column=0)
ca_entry = tk.Entry(root)
ca_entry.grid(row=11, column=1)

tk.Label(root, text="Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversable Defect)").grid(row=12, column=0)
thal_entry = tk.Entry(root)
thal_entry.grid(row=12, column=1)

# Create and place the Predict button
predict_button = tk.Button(root, text="Predict", command=predict_heart_disease)
predict_button.grid(row=13, column=0, columnspan=2)

# Start the GUI event loop
root.mainloop()
