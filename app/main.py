from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    df = pd.DataFrame([data], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = model.predict(df)[0]
    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)