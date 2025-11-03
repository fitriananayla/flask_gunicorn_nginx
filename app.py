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


# ====================================================
# === DASHBOARD DENGAN DASH (DATA BUAH)
# ====================================================

dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Dataset Buah
df = pd.DataFrame({
    'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
    'Amount': [4, 1, 2, 2, 4, 5],
    'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
})

# --- AGREGASI DATA ---
df_sum = df.groupby(['Fruit', 'City'], as_index=False)['Amount'].sum()

# --- GRAFIK BAR ---
fig = px.bar(
    df_sum,
    x='Fruit',
    y='Amount',
    color='City',
    barmode='group',
    title='Jumlah Buah Berdasarkan Kota'
)

fig.update_layout(
    xaxis_title="Jenis Buah",
    yaxis_title="Jumlah",
    title_x=0.5,
    plot_bgcolor='rgba(245,245,245,0.8)',
    paper_bgcolor='rgba(255,255,255,1)',
    font=dict(size=14)
)

# --- LAYOUT DASHBOARD ---
dash_app.layout = html.Div([
    html.H1("Dashboard Data Buah", style={
        'textAlign': 'center',
        'color': '#222',
        'marginBottom': 30
    }),

    html.H3("Tabel Data Buah:", style={'marginLeft': 30}),
    dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=6,
        style_table={'overflowX': 'auto', 'marginLeft': 30, 'marginRight': 30},
        style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 14},
        style_header={'backgroundColor': '#EAEAEA', 'fontWeight': 'bold'}
    ),

    html.H3("Grafik Jumlah Buah per Kota:", style={'marginLeft': 30, 'marginTop': 40}),
    dcc.Graph(figure=fig),

    html.Div([
        html.A("⬅️ Kembali ke Halaman Utama", href='/', style={
            'display': 'block',
            'textAlign': 'center',
            'marginTop': 40,
            'fontSize': '18px',
            'textDecoration': 'none',
            'color': '#007BFF'
        })
    ])
])


# ====================================================
# === JALANKAN APLIKASI FLASK
# ====================================================
if __name__ == '__main__':
    app.run(debug=True)
