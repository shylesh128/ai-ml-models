import os
import tempfile

from flask import Flask, render_template, request, jsonify
from keras import models
from PIL import Image
import numpy as np

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

model = models.load_model('ai-models/image-prediction.keras')
app = Flask(__name__, template_folder='templates')

def predict_image(model, path_to_image):
    img = Image.open(path_to_image)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asanyarray(img)
    data = data/255
    print(data[0][0], "after")
    probs = model.predict(np.array([data])[:1])
    print(probs)
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    print(top_prob, top_pred)
    return top_prob, top_pred


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the image prediction logic here
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    uploaded_file = request.files['image']

    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.filename)
        uploaded_file.save(temp_path)
        percentage, result = predict_image(model, temp_path)

        

        # Convert float to a Python float
        percentage = float(percentage)
        print(result)

        # Create a dictionary with standard Python data types
        response_data = {'prediction': percentage, 'result': result}
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        # Clean up: remove the temporary directory and its contents
        if os.path.exists(temp_dir):
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error cleaning up {file_path}: {e}")

            os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)
