import pandas as pd
import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('./data/traffic_data.csv')

X = df.drop('traffic_condition', axis=1)
y = df['traffic_condition']

print(f"Unique traffic conditions: {y.unique()}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

all_labels = sorted(y.unique())

train_report = classification_report(y_train, y_train_pred, labels=all_labels, output_dict=False)
val_report = classification_report(y_val, y_val_pred, labels=all_labels, output_dict=False)
test_report = classification_report(y_test, y_test_pred, labels=all_labels, output_dict=False)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

with open('./src/perceptron.txt', 'w') as report_file:
    report_file.write("Training Report:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       Total samples: {len(y_train)}\n")
    
    report_file.write("Validation Report:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       Total samples: {len(y_val)}\n")
    
    report_file.write("Testing Report:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       Total samples: {len(y_test)}\n")

print(f'Perceptron Training Accuracy: {train_accuracy:.2f}')
print(f'Perceptron Validation Accuracy: {val_accuracy:.2f}')
print(f'Perceptron Testing Accuracy: {test_accuracy:.2f}')

output_dir = 'D:/machine learning/traffic_prediction/web/static/png'
os.makedirs(output_dir, exist_ok=True)

conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')

confusion_matrix_image_path = os.path.join(output_dir, 'perceptron_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()

print(f"Ma trận nhầm lẫn đã được lưu tại: {confusion_matrix_image_path}")

train_sizes = [100, 200, 300, 400, 500]
train_scores = []
val_scores = []

for train_size in train_sizes:
    X_train_subset = X_train[:train_size]
    y_train_subset = y_train[:train_size]
    
    model.fit(X_train_subset, y_train_subset)
    train_scores.append(model.score(X_train_subset, y_train_subset))
    val_scores.append(model.score(X_val, y_val))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Training score', color='blue')
plt.plot(train_sizes, val_scores, label='Validation score', color='orange')
plt.title('Learning Curve')
plt.xlabel('Training Size (Number of Samples)')
plt.ylabel('Score')
plt.legend()

learning_curve_image_path = os.path.join(output_dir, 'perceptron_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()

print(f"Learning curve đã được lưu tại: {learning_curve_image_path}")

with open('./src/perceptron_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Perceptron model saved successfully!")


















