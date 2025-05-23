import pickle
import numpy as np
from sklearn.datasets import load_iris
from flask import Flask, render_template, request

app = Flask(__name__)

with open("iris_classifier.bin", 'rb') as f_in:
    model, iris_data = pickle.load(f_in)


iris_dict = {
    "setosa": "iris_setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}


@app.route('/')
def index():
    iris = load_iris(as_frame=True)
    df = iris.frame

    df['target'] = iris.target
    df['class'] = df['target'].map(dict(enumerate(iris.target_names)))

    summary = df.groupby('class').agg(['min', 'max']).drop("target", axis=1)
    summary_html = summary.to_html(classes="table table-bordered", border=0)

    return render_template('index.html', summary_table=summary_html)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        prediction_class_name = iris_data.target_names[int(prediction[0])]

        prediction_image = iris_dict[prediction_class_name]

        return render_template('result.html', prediction=prediction_class_name, image=prediction_image)

    except ValueError:
        print("Invalid Input. Please Enter Numeric Value")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9555)