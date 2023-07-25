from flask import Flask, request, jsonify, render_template
import pickle
import numpy as  np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Pregnancies = float(request.form["pregnancies"])
    Glucose = float(request.form["glucose"])
    BloodPressure = float(request.form["bloodPressure"])
    SkinThickness = float(request.form["skinThickness"])
    Insulin = float(request.form["insulin"])
    BMI = float(request.form["bmi"])
    DiabetesPedigreeFunction = float(request.form["diabetesPedigreeFunction"])
    Age = float(request.form["age"])

    # Bentuk array fitur untuk prediksi
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin ,BMI , DiabetesPedigreeFunction,Age ]])

    # Lakukan prediksi menggunakan model
    prediction = model.predict(features)
    if(prediction == 0):
        return render_template("index.html", prediction_text = "Negatif")
    elif(prediction == 1):
        return render_template("index.html", prediction_text = "Positif")
    

if __name__ == "__main__":
    app.run(debug=True)