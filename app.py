
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# تحميل النموذج من رابط خارجي
MODEL_URL = "https://www.dl.dropboxusercontent.com/scl/fi/bx919lqd3r3yvtsxi50ns/stone_classifier_model.h5?rlkey=q4xnvaiw1lm61xr5xcqbcoppy&st=7jqoakb2&dl=1"
MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)

download_model()
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['يشب', 'ياقوت', 'جمشت', 'زمرد', 'فيروز', 'عقيق', 'اللؤلؤ الأبيض', 'اللؤلؤ الأسود']

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            label = class_names[class_index]
            return render_template('index.html', label=label, image_url=filepath)

    return render_template('index.html', label=None, image_url=None)

if __name__ == '__main__':
    app.run(debug=True)
