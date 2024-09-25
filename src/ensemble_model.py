import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

df = pd.read_csv('./data/traffic_data.csv')

X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

base_models = [
    ('perceptron', Perceptron(max_iter=1000, random_state=42)),
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('neural_network', MLPClassifier(random_state=42, max_iter=1000))
]

meta_model = LogisticRegression()

model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_report = classification_report(y_train, y_train_pred, output_dict=False)
val_report = classification_report(y_val, y_val_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

with open('./src/ensemble.txt', 'w') as report_file:
    report_file.write("Training:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       {len(y_test)}\n")

print(f'Ensemble Model Training Accuracy: {train_accuracy:.2f}')
print(f'Ensemble Model Validation Accuracy: {val_accuracy:.2f}')
print(f'Ensemble Model Testing Accuracy: {test_accuracy:.2f}')

try:
    with open('./src/ensemble_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Ensemble model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")








