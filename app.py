from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# import pandas as pd
# import datetime

app = Flask(__name__)
Load_model = load_model("Monthly_Model.h5")
Scaler = pickle.load(open('MinmaxScaler.pkl', 'rb'))


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('index.html')


@app.route('/method', methods=['POST'])
def predict():
    Value = request.form["Close Price"]
    scaled = Scaler.transform(np.array(Value).reshape(-1, 1))
    prediction = Load_model.predict(scaled.reshape(scaled.shape[0], 1, scaled.shape[1]))
    Result = Scaler.inverse_transform(prediction)[0][0]
    return render_template("result.html", result=Result)


if __name__ == "__main__":
    app.run(debug=True)
