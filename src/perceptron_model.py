import pandas as pd
import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/traffic_data.csv')


X = df.drop('traffic_condition', axis=1)  
y = df['traffic_condition']  


print(f"Unique traffic conditions: {y.unique()}")


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 40% train + validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 20% train, 20% validation


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


param_grid = {
    'max_iter': [1000, 2000],
    'eta0': [0.01, 0.1, 1.0],
    'tol': [1e-4, 1e-3]
}
grid_search = GridSearchCV(Perceptron(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_report = classification_report(y_train, y_train_pred, output_dict=False)
val_report = classification_report(y_val, y_val_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)


train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


cv_scores = cross_val_score(model, X_train, y_train, cv=5)


with open('./src/perceptron.txt', 'w') as report_file:
    report_file.write("Training:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       {len(y_test)}\n")
    
    report_file.write(f"\nCross-validation scores (5-fold): {cv_scores}\n")
    report_file.write(f"Mean CV accuracy: {cv_scores.mean():.2f}\n")

print(f'Perceptron Training Accuracy: {train_accuracy:.2f}')
print(f'Perceptron Validation Accuracy: {val_accuracy:.2f}')
print(f'Perceptron Testing Accuracy: {test_accuracy:.2f}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}')


with open('./src/perceptron_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Perceptron model saved successfully!")











