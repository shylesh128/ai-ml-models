
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model


model = load_model('ai-models/chart-text-class-m2.h5')
model.summary()

class_names = {
    'animal': 0,
    'chart': 1,
    'human': 2,
    'objects': 3,
    'scenery': 4,
    'text': 5,
    'vehicle': 6
}

# app = Flask(__name__, template_folder='templates')
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from the request
        file = request.files['image']
        
        # Read and preprocess the image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        resize = tf.image.resize(img, (256, 256))
        img_array = np.expand_dims(resize / 255, 0)

        # Make predictions
        yhat = model.predict(img_array)
        predicted_index = np.argmax(yhat)
        predicted_class = [key for key, value in class_names.items() if value == predicted_index][0]
        yhat_list = yhat.tolist()

        # Prepare response
        response_data = {
            "predicted_class": predicted_class,
            "predicted_percentage": yhat_list[0][predicted_index],
            "all_predicted_classes": {class_name: yhat_list[0][index] for class_name, index in class_names.items()}
        }

        

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
