from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form['model']
    data = {
        'Pregnancies': int(request.form['pregnancies']),
        'Glucose': int(request.form['glucose']),
        'BloodPressure': int(request.form['blood_pressure']),
        'SkinThickness': int(request.form['skin_thickness']),
        'Insulin': int(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
        'Age': int(request.form['age'])
    }

    df = pd.DataFrame(data, index=[0])
    df_scaled = scaler.transform(df)

    selected_model_idx = model_names.index(selected_model)
    selected_model_obj = model[selected_model_idx]

    prediction = selected_model_obj.predict(df_scaled)
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

    return render_template('index.html', model_names=model_names, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
