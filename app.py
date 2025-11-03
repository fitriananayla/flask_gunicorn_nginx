from flask import Flask, render_template, request
import pickle
import pandas as pd
from dash import Dash, html, dcc, dash_table
import plotly.express as px

# --- INISIALISASI FLASK ---
app = Flask(__name__)

# --- LOAD MODEL MACHINE LEARNING ---
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

# --- HALAMAN UTAMA FLASK ---
@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

# --- HALAMAN PREDIKSI ---
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

# Dataset dari file PDF
df = pd.DataFrame({
    'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
    'Amount': [4, 1, 2, 2, 4, 5],
    'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
})

# Buat grafik
fig = px.bar(
    df,
    x='Fruit',
    y='Amount',
    color='City',
    barmode='group',
    title='Jumlah Buah Berdasarkan Kota'
)

# Layout dashboard
dash_app.layout = html.Div([
    html.H1("Dashboard Data Buah", style={
        'textAlign': 'center',
        'color': '#333',
        'marginBottom': 30
    }),
    html.H3("Tabel Data Buah:", style={'marginTop': 20}),
    dash_table.DataTable(
        data=df.to_dict('records'),
        page_size=6,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'}
    ),
    html.H3("Grafik Jumlah Buah per Kota:", style={'marginTop': 30}),
    dcc.Graph(figure=fig)
])


# --- JALANKAN FLASK ---
if __name__ == '__main__':
    app.run(debug=True)
