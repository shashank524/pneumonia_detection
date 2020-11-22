from flask import Flask, render_template, request
from keras.applications.resnet50 import *
from keras.applications.inception_v3 import *
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
import os
import json

app = Flask(__name__, template_folder=os.path.abspath("static/"))
global graph
global sess
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = load_model('chest-xray-pneumonia.h5')
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
UPLOAD_FOLDER = 'static/'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(path):
    img = image.load_img(path, target_size=(224,224))
    xy = image.img_to_array(img)
    xy = np.expand_dims(xy, axis=0)
    xy = preprocess_input(xy)

    with graph.as_default():
        set_session(sess)
        preds = model.predict(xy)
        # preds_decoded = decode_predictions(preds, top=3)[0]
        print(preds)
        acc = []
        classes = []
    for x in preds:
        acc.append(x[1]*100)
        acc.append((1-x[1])*100)
        classes.append('pnemonia')
        classes.append('normal')
    return acc, classes


@app.route("/")
def index():
	# return the rendered template
	return render_template('gui.html')


@app.route("/detect", methods=["GET", "POST"])
def detect():

    if request.method == "GET":
        return render_template("object.html")

    if request.method == "POST":
        image = request.files["x-ray"]
        image_name = image.filename
        path = os.path.join(UPLOAD_FOLDER, image_name)
        image.save(path)
        # path = request.form["path"]':


        # accuracies, classes = predict(path)
        accuracies, classes = predict(path)
        os.remove(path)
	    
        return render_template("object.html", preds=accuracies,
                               classes=json.dumps(classes), img=path)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
