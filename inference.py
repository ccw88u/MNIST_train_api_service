import base64
from io import BytesIO
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('best_model.h5')

# Define the Flask app
app = Flask(__name__)

# Define a function to preprocess the image
def preprocess_image(image_b64):
    # Convert the base64-encoded image to a PIL Image object
    image_bytes = BytesIO(base64.b64decode(image_b64))
    image = Image.open(image_bytes)
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image) / 255.
    image_tensor = image_array.reshape((1, 28, 28, 1))
    return image_tensor

@app.route('/predict', methods=['POST'])
def predict_digit():
    image_b64 = request.json['image']
    image = preprocess_image(image_b64)
    prediction = model.predict(image)
    print('prediction:', prediction)
    digit = np.argmax(prediction)
    response = {'digit': str(digit)}
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)