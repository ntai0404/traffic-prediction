import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/traffic_data.csv')
df['time_of_day'] = df['time_of_day'].astype(int)

X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [500, 1000],
    'early_stopping': [True],
    'validation_fraction': [0.1],
}

mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_report = classification_report(y_train, y_train_pred, output_dict=False)
val_report = classification_report(y_val, y_val_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

with open('./src/neural_network.txt', 'w') as report_file:
    report_file.write("Training:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       {len(y_test)}\n")

print(f'Neural Network Training Accuracy: {train_accuracy:.2f}')
print(f'Neural Network Validation Accuracy: {val_accuracy:.2f}')
print(f'Neural Network Testing Accuracy: {test_accuracy:.2f}')

with open('./src/neural_network_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Neural Network model saved successfully!")

















