import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

output_dir = 'D:/machine learning/traffic_prediction/web/static/png'
os.makedirs(output_dir, exist_ok=True)

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
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [500, 1000],
    'learning_rate_init': [0.001, 0.01],
    'early_stopping': [True],
    'validation_fraction': [0.1],
    'batch_size': ['auto', 32, 64],
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
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       Total samples: {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       Total samples: {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       Total samples: {len(y_test)}\n")

print(f'Neural Network Training Accuracy: {train_accuracy:.2f}')
print(f'Neural Network Validation Accuracy: {val_accuracy:.2f}')
print(f'Neural Network Testing Accuracy: {test_accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')

confusion_matrix_image_path = os.path.join(output_dir, 'neural_network_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=3, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')

learning_curve_image_path = os.path.join(output_dir, 'neural_network_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()

with open('./src/neural_network_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Neural Network model saved successfully!")
