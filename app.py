from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# تحميل النموذج
model = load_model("model.h5")

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