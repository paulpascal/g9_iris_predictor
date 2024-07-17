from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        df = pd.DataFrame(
            [data],
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        prediction = model.predict(df)[0]
        return jsonify(result={"prediction": str(prediction)})
    except Exception as e:
        print(e)
        return jsonify(error={"message": "An error has occurred"})

