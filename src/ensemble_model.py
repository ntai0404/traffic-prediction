import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

output_dir = 'D:/machine learning/traffic_prediction/web/static/png'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('./data/traffic_data.csv')

X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

with open('./src/perceptron_model.pkl', 'rb') as file:
    perceptron_model = pickle.load(file)

with open('./src/id3_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('./src/neural_network_model.pkl', 'rb') as file:
    neural_network_model = pickle.load(file)

base_models = [
    ('perceptron', perceptron_model),
    ('decision_tree', decision_tree_model),
    ('neural_network', neural_network_model)
]

meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
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

conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - Ensemble Model')

confusion_matrix_image_path = os.path.join(output_dir, 'ensemble_model_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()
print(f"Confusion matrix saved at: {confusion_matrix_image_path}")

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=3, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve - Ensemble Model')
plt.legend(loc='best')

learning_curve_image_path = os.path.join(output_dir, 'ensemble_model_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()
print(f"Learning curve saved at: {learning_curve_image_path}")

try:
    with open('./src/ensemble_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Ensemble model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")











