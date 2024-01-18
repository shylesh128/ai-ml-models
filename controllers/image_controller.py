# controllers/image_controller.py
import os
import tempfile
from flask import jsonify, request, render_template
from models.image_prediction_model import load_image_prediction_model, predict_image
from models.image_classifer_chart_model import load_image_classifier_model, predict_classes

first_image_model = load_image_prediction_model()
model = load_image_classifier_model()

def index():
    return render_template('index.html')

def classes():
    return render_template('class.html')

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
        percentage, result = predict_image(first_image_model, temp_path)

        

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

def predict_image_classes():
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
        # percentage, result = predict_classes(first_image_model, temp_path)
        predict_classes(first_image_model, temp_path)

        # percentage = float(percentage)
        # response_data = {'prediction': percentage, 'result': result}
        # return jsonify(response_data)
    
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