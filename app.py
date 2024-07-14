from flask import Flask, render_template
from flask import url_for, request, redirect
from flask import jsonify
from flask import make_response
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import keras
import tensorflow as tf

app = Flask(__name__)


predict_value = "Predict: "
model = keras.models.load_model("models/model.h5")


@app.route("/")
@app.route("/home")
def home():
    my_variable = request.cookies.get('predict_value', predict_value)
    response = make_response(render_template('index.html'))
    response.set_cookie('my_variable', 'specific_value')
    return response


@app.route("/predict", methods=["POST"])
def predict():
    print("received")
    data = request.json
    image = data.get("image")
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    img = np.array(image)[:, :, 0]
    img = cv2.resize(img, (28, 28))
    top, bottom, left, right = 4, 4, 4, 4
    img = cv2.copyMakeBorder(img,
                             top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = tf.expand_dims(img, axis=2)
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img).argmax(axis=1)

    return jsonify({
        "message": predict_value + f"{prediction[0]}"
    })


if __name__ == "__main__":
    app.run(debug=True)
