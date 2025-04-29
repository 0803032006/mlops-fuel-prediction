# app.py
from flask import Flask, render_template, request
import joblib
import datetime

app = Flask(__name__)

# Load model and label encoder
model = joblib.load('svr_model.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    fuel_type = request.form['fuel_type']

    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").toordinal()
        fuel_type_encoded = le.transform([fuel_type])[0]

        prediction = model.predict([[date, fuel_type_encoded]])[0]
        prediction = round(prediction, 2)

        return render_template('result.html', prediction_text=f"Predicted Price: ₹{prediction}")
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
