from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img, img_to_array
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)

dic = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}

model = load_model('model.h5')
model.make_predict_function()

archive_data = []

def predict_label(image_path):
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return dic[predicted_class]

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST' and 'imagefile' in request.files:
        imagefile = request.files['imagefile']
        
        image_path = "./uploads/" + imagefile.filename
        imagefile.save(image_path)
        p = predict_label(image_path)
        
        # convert IMG to Bytecode
        im = Image.open(image_path)
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        
        archive_data.append({'image_path': (encoded_img_data.decode('utf-8')), 'prediction': p})       
        
        
        return render_template("index.html", prediction=p, image_path=encoded_img_data.decode('utf-8'))

@app.route('/archive', methods=['GET'])
def archive():
    return render_template("archive.html", archive_data=archive_data)

if __name__ == '__main__':
    app.run(port=5500, debug=True)
