import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg cho việc lưu hình ảnh mà không cần hiển thị

from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import os

app = Flask(__name__, static_folder='D:/machine learning/traffic_prediction/web/static')

# Load the models
models = {
    'Perceptron': pickle.load(open('./src/perceptron_model.pkl', 'rb')),
    'ID3': pickle.load(open('./src/id3_model.pkl', 'rb')),
    'Neural Network': pickle.load(open('./src/neural_network_model.pkl', 'rb')),
    'Ensemble Model': pickle.load(open('./src/ensemble_model.pkl', 'rb'))
}

# Function to read report from text file
def read_report(model_name):
    try:
        if model_name.lower() == "ensemble model":
            with open('./src/ensemble.txt', 'r') as file:
                report = file.read()
        else:
            with open(f'./src/{model_name.lower().replace(" ", "_")}.txt', 'r') as file:
                report = file.read()
        return report
    except FileNotFoundError as e:
        return "Report not found.", e

# Function to create confusion matrix
def create_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    cm_percentage = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            cm_percentage[i] = cm[i] / cm[i].sum() * 100

    cm_display = np.empty((cm.shape[0], cm.shape[1]), dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_display[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=cm_display, fmt='', cmap='Blues', cbar=False, 
                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], 
                linewidths=0.5, linecolor='black')

    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    image_path = 'D:/machine learning/traffic_prediction/web/static/confusion_matrix.png'
    plt.savefig(image_path)
    plt.close()

# Function to create learning curve
def create_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training score', color='blue')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', color='green')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    
    image_path = 'D:/machine learning/traffic_prediction/web/static/learning_curve.png'
    plt.savefig(image_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html', models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form.get('model')
    
    # Load data from CSV
    df = pd.read_csv('./data/traffic_data.csv')
    X = df.drop('traffic_condition', axis=1)
    y = df['traffic_condition']

    # Load the selected model
    model = models[selected_model]
    
    # Predict using the whole dataset
    predictions = model.predict(X)

    # Create confusion matrix
    create_confusion_matrix(y, predictions, selected_model)
    
    # Create learning curve
    create_learning_curve(model, X, y, selected_model)

    # Read the report for the selected model
    report = read_report(selected_model)
    
    return render_template('index.html', 
                           prediction=predictions[0],  # Example, modify as needed
                           models=models.keys(), 
                           selected_model=selected_model, 
                           report=report,
                           confusion_matrix_image='/static/confusion_matrix.png',
                           learning_curve_image='/static/learning_curve.png')  # Path to learning curve image

if __name__ == '__main__':
    app.run(debug=True)











