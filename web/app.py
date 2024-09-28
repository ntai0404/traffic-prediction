import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder='D:/machine learning/traffic_prediction/web/static')


models = {
    'Perceptron': pickle.load(open('./src/perceptron_model.pkl', 'rb')),
    'ID3': pickle.load(open('./src/id3_model.pkl', 'rb')),
    'Neural Network': pickle.load(open('./src/neural_network_model.pkl', 'rb')),
    'Ensemble Model': pickle.load(open('./src/ensemble_model.pkl', 'rb'))
}

def read_report(model_name):
    try:
        if model_name.lower() == "ensemble model":
            with open('./src/ensemble.txt', 'r') as file:
                report = file.read()
        else:
            with open(f'./src/{model_name.lower().replace(" ", "_")}.txt', 'r') as file:
                report = file.read()
        return report
    except FileNotFoundError:
        return "Report not found."

@app.route('/')
def index():
    return render_template('index.html', models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form.get('model')
    
 
    df = pd.read_csv('./data/traffic_data.csv')
    X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
    y = df['traffic_condition']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = models[selected_model]
    
  
    predictions = model.predict(X_test)

    report = read_report(selected_model)
    
   
    confusion_matrix_image = f'/static/png/{selected_model.lower().replace(" ", "_")}_confusion_matrix.png'
    learning_curve_image = f'/static/png/{selected_model.lower().replace(" ", "_")}_learning_curve.png'

    return render_template('index.html', 
                           prediction=predictions[0],  
                           models=models.keys(), 
                           selected_model=selected_model, 
                           report=report,
                           confusion_matrix_image=confusion_matrix_image,
                           learning_curve_image=learning_curve_image)

if __name__ == '__main__':
    app.run(debug=True)



