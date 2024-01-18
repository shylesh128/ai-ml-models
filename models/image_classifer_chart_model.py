# models/image_prediction_model.py
from keras import models
from PIL import Image
import numpy as np

from keras.models import load_model

class_names = {
    'animal': 0,
    'chart': 1,
    'human': 2,
    'objects': 3,
    'scenery': 4,
    'text': 5,
    'vehicle': 6
}

def load_image_classifier_model():
    return load_model('ai-models/chart-classifier-m1.keras')


def predict_classes(model, path_to_image):
    print(model.summary())
    # img = Image.open(path_to_image)
    # img = img.convert("RGB")
    # img = img.resize((32,32))
    # data = np.asanyarray(img)
    # data = data/255
    # probs = model.predict(np.array([data])[:1])
    # top_prob = probs.max()
    # top_pred = class_names[np.argmax(probs)]
    # print(top_prob, top_pred)
    # return top_prob, top_pred
    