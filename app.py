from flask import Flask, render_template, request
import pickle
import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px

app = Flask(__name__)

# --- LOAD MODEL ---
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    model_names = ['Decision Tree', 'SVC']
except:
    model = None
    scaler = None
    model_names = ['Model Belum Tersedia']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("index.html", prediction="Model belum di-load.")
    
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

    input_data = pd.DataFrame(data, index=[0])
    input_data_scaled = scaler.transform(input_data)

    selected_model_idx = model_names.index(selected_model)
    selected_model_obj = model[selected_model_idx]
    prediction = selected_model_obj.predict(input_data_scaled)
    prediction = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

    return render_template('index.html', model_names=model_names, prediction=prediction)

# --- DASHBOARD DENGAN DASH ---
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
fig = px.bar(df, x='continent', y='lifeExp', color='continent', barmode='group')

dash_app.layout = html.Div([
    html.H1("Visualisasi Data Global", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run(debug=True)
