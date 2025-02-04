from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model  # Import for loading .h5 models
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

# Load your new model in .h5 format
model_name = 'Model.h5'  # Change this to your new .h5 model's filename

try:
    model = load_model(model_name)  # Load the Keras model
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Image size and class mapping based on your new model
IMAGE_SIZE = (128, 128)  # Resize all images to 128x128
CLASS_MAPPING = {'A': 0, 'D': 1, 'N': 2, 'G': 3}
CLASS_NAMES = ['Age related Macular Degenration', 'Diabetic Retinopathy', 'Normal', 'Glaucoma']  # List of classes corresponds to CLASS_MAPPING

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream)  # Open the uploaded image file
        img = img.resize(IMAGE_SIZE)  # Resize to model input size (128x128)
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = img_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)  # Adjust dimensions for input

        prediction = model.predict(img_array)  # Make a prediction
        result = np.argmax(prediction, axis=-1)[0]  # Get index of the highest score

        label = CLASS_NAMES[result]  # Get the corresponding label using CLASS_NAMES

        return jsonify({'prediction': label})  # Return the class label
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error details

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
