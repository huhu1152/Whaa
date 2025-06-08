from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# رابط تحميل النموذج
MODEL_URL = "https://www.dl.dropboxusercontent.com/scl/fi/bx919lqd3r3yvtsxi50ns/stone_classifier_model.h5?dl=1"
MODEL_PATH = "model.h5"

# تحميل النموذج لو غير موجود
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['يشب', 'ياقوت', 'جمشت', 'زمرد', 'فيروز', 'عقيق', 'اللؤلؤ الأبيض', 'اللؤلؤ الأسود']

def preprocess_image(path):
    img = Image.open(path).convert('RGB').resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, 0)

@app.route('/', methods=['GET','POST'])
def index():
    label = None
    image_url = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_arr = preprocess_image(filepath)
            pred = model.predict(img_arr)
            idx = np.argmax(pred)
            label = class_names[idx]
            image_url = url_for('static', filename=f"uploads/{filename}")
    return render_template('index.html', label=label, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
