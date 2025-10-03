from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load("iris_model.pkl")
accuracy = joblib.load("model_accuracy.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get form data
        sepal_length = float(request.form["sepal_length"])
        sepal_width  = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width  = float(request.form["petal_width"])

        # Make prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)[0]
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        prediction = species_map[pred]

    return render_template("index.html", prediction=prediction, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
