from flask import Flask, render_template, request
import joblib
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and accuracy
model = joblib.load("iris_model.pkl")
accuracy = joblib.load("model_accuracy.pkl")  # This is a float number like 0.95

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        features = [[
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]]
        prediction_class = model.predict(features)[0]
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        prediction = species_map[prediction_class]

    return render_template("index.html", prediction=prediction, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)

