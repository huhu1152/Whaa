from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# تحميل النموذج

import requests

model_path = "downloaded_model.h5"
if not os.path.exists(model_path):
    print("Downloading model from Dropbox...")
    with requests.get("https://www.dropbox.com/scl/fi/bx919lqd3r3yvtsxi50ns/stone_classifier_model.h5?rlkey=q4xnvaiw1lm61xr5xcqbcoppy&st=7jqoakb2&dl=1", stream=True) as r:
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

model = load_model(model_path)


# أسماء الفئات - قم بتعديلها حسب نموذجك
class_names = ["class_1", "class_2", "class_3"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = round(100 * np.max(prediction), 2)
            result = f"Prediction: {predicted_class} ({confidence}%)"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)