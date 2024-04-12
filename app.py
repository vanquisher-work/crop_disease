from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)

# Load class indices mapping
class_mapping_file = os.path.join(os.path.dirname(__file__), 'class_indices.json')
with open(class_mapping_file, 'r') as f:
    class_mapping = json.load(f)

# Load TensorFlow Lite model
model_file = os.path.join(os.path.dirname(__file__), 'model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def process_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array.astype('float32') / 255  # Normalize pixel values to [0, 1]

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])
    predicted_label = [label for label, index in class_mapping.items() if index == predicted_class]

    if predicted_label:
        return predicted_label[0]
    else:
        return "Invalid predicted index."

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    crop_name = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No image found in request'
        else:
            image_file = request.files['image']
            if image_file.filename == '':
                error = 'No image selected for uploading'
            else:
                image_path = 'temp_image.jpg'  # Save uploaded image temporarily
                image_file.save(image_path)
                predicted_label = process_image(image_path)
                crop_name, status = predicted_label.split('___')
                if status == 'healthy':
                    prediction = f"Crop Name: {crop_name.capitalize()}. \nYour crop is healthy or disease-free."
                else:
                    prediction = f"Crop Name: {crop_name.capitalize()}. \nYour crop has the following disease: {status}."
                os.remove(image_path)   # Delete temporary image file after prediction

    return render_template('index.html', prediction=prediction, error=error,crop_name=crop_name)

if __name__ == '__main__':
    app.run(debug=True)
