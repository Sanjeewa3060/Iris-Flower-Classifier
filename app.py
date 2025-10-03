from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and accuracy
model = joblib.load("iris_model.pkl")
accuracy = joblib.load("model_accuracy.pkl")  # Float value like 0.9733

# Mapping if model predicts numeric labels
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred_class = model.predict(features)[0]

            if isinstance(pred_class, int):
                prediction = species_map.get(pred_class, "Unknown")
            else:
                prediction = str(pred_class)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, accuracy=accuracy*100)

if __name__ == "__main__":
    app.run(debug=True)
