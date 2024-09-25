import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('./data/traffic_data.csv')

# Ensure time_of_day is in the correct format (0-3)
df['time_of_day'] = df['time_of_day'].astype(int)

# Features and target (updated to include all relevant features)
X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

# Split the dataset into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.6, random_state=42)  # 60% test, 40% temp
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% train, 20% validation

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],  # None means nodes are expanded until all leaves are pure
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the model and perform grid search
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
grid_search = GridSearchCV(dtc, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
model = grid_search.best_estimator_

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
with open('./src/id3.txt', 'w') as report_file:
    report_file.write("Training:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       {len(y_test)}\n")

# Print accuracy for all three datasets
print(f'ID3 Training Accuracy: {train_accuracy:.2f}')
print(f'ID3 Validation Accuracy: {val_accuracy:.2f}')
print(f'ID3 Testing Accuracy: {test_accuracy:.2f}')

# Save the model
with open('./src/id3_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("ID3 model saved successfully!")
