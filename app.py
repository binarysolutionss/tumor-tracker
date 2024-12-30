import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #for floating point representation issue with pc

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import model_from_json

model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

# Custom objects dictionary with lowercase
custom_objects = {
    'maxpooling2d': MaxPooling2D,
    'MaxPooling2D': MaxPooling2D,
    'maxpooling_2d': MaxPooling2D,
    'max_pooling2d': MaxPooling2D,
    'max_pooling_2d': MaxPooling2D
}

# Create a Flask instance
app=Flask(__name__)

# try:
#     # Load model architecture from json
#     with open('models/model_architecture.json', 'r') as json_file:
#         model_json = json_file.read()
        
#     # Create model from JSON
#     model = model_from_json(model_json)
    
#     # Load weights
#     model.load_weights('models/model_weights.weights.h5')
    
#     # Compile model
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     print("Model loaded successfully!")
    
# except Exception as e:
#     print(f"Error loading model: {e}")

# # Load the mdoel
try:
    model = load_model('models/melanoma_model.h5', custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    
    # Add fallback model loading
    try: 
        model = tf.keras.models.load_model('models/melanoma_model.h5', compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Model loaded with fallback nethod!")
    except Exception as e:
        print(f"Fallback loading failed: {e}")
    
melanoma_info = {
    'positive':{
        'description': 'Potential melanoma detected',
        'recommendations':[
            'Consult a dematologist immediately',
            'Take photos of the lesion for documentatio npurposes',
            'Do not wait or delay seeking professional medical attention',
            'Prepare a list of any symptoms or changes you\'ve noticed'
        ]
    },
    'negative':{
        'description': 'No melanoma indicators detected',
        'recommendations':[
            'Continue regular skin self-examinations',
            'Use sun protection',
            'Monitor any changes in your skin',
            'Schedule routine skin check-ups with your doctor'
        ]
    },
}

def process_image(image_file):
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file uploaded')
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('index.html', error='Invalid file format')
        
        img_array = process_image(file)
        prediction = model.predict(img_array)
        
        result='positive' if prediction[0][0] > 0.5 else 'negative'
        confidence=float(prediction[0][0]) if result == 'positive' else float(1 - prediction[0][0])
        
        return render_template('index.html',
                            prediction=result,
                            confidence=f"{confidence*100:.2f}%",
                            info=melanoma_info[result])
        
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {str(e)}')
    
if __name__ == '__main__':
    app.run(debug=True)