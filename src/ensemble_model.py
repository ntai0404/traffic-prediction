import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('./data/traffic_data.csv')

# Features and target (only keep the 5 important features)
X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles']]
y = df['traffic_condition']

# Split the dataset into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Create and train the ensemble model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Set random_state for reproducibility
model.fit(X_train, y_train)

# Predict and evaluate
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Generate classification reports
train_report = classification_report(y_train, y_train_pred, output_dict=False)
val_report = classification_report(y_val, y_val_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Save reports to a text file
with open('./src/ensemble.txt', 'w') as report_file:
    report_file.write("Training:\n")
    report_file.write(train_report)
    report_file.write(f"\naccuracy                           {train_accuracy:.2f}       {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\naccuracy                           {val_accuracy:.2f}       {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\naccuracy                           {test_accuracy:.2f}       {len(y_test)}\n")

# Print accuracy for all three datasets
print(f'Ensemble Model (Random Forest) Training Accuracy: {train_accuracy:.2f}')
print(f'Ensemble Model (Random Forest) Validation Accuracy: {val_accuracy:.2f}')
print(f'Ensemble Model (Random Forest) Testing Accuracy: {test_accuracy:.2f}')

# Save the model
try:
    with open('./src/ensemble_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Ensemble model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")




