# models/image_prediction_model.py
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

def load_image_prediction_model():
    return models.load_model('ai-models/image-prediction.keras')


def predict_image(model, path_to_image):
    img = Image.open(path_to_image)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asanyarray(img)
    data = data/255
    probs = model.predict(np.array([data])[:1])
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    print(top_prob, top_pred)
    return top_prob, top_pred
    