## TumorTracker

![tumor tracker](https://github.com/user-attachments/assets/5993c7db-3da3-4075-bb8e-bc722f3bd1e2)

A machine learning-powered web application for melanoma skin cancer detection using deep learning and computer vision.

## Overview

TumorTracker is a Flask-based web application that uses a convolutional neural network (CNN) to analyze images of skin lesions and detect potential melanoma. The system provides real-time analysis with confidence scores and recommended actions based on the detection results.

## Features

- Real-time image processing and analysis
- Deep learning-based melanoma detection
- Confidence score reporting
- Custom recommendations based on detection results
- User-friendly web interface
- Support for multiple image formats (PNG, JPG, JPEG)

## Technology Stack

- Backend Framework: Flask (Pyhthon)
- Machine Learning: TensorFlow/Keras
- Frontend: HTML, CSS, JavaScript
- Image Processing: PIL (Python Imaging Library)
- Deep Learning Model: Custom CNN Architecture

## Model Architecture

The CNN model consists of:

- 3 Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with dropout for regularization
- Binary classification output

## Installation

Download melanoma test data images
"https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"

## Clone the repository:

```
bashCopygit clone [repository-url]
```

## Install required dependencies:

- Flask
- Tensorflow
- numpy

## Run the application:

```
bashCopypython app.py
```

## Usage

1. Navigate to the web interface (default: http://localhost:5000)
2. Upload an image of the skin lesion
3. Click "Upload" to process the image
4. View the detection results and recommendations

## Training the Model

To train the model with your own dataset:

Organize your dataset in the following structure:

data/melanoma/  
├── train/  
│   ├── benign/  
│   └── malignant/  
└── test/  
├── benign/  
    └── malignant/

## Run the training script:

```
bashCopypython train_model.py
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This tool is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical advice.

##Developed by Binary Solutions

Thanks to all contributors and the open-source community

## Contact

For support or queries, please contact Binary Solutions through the provided contact information in the application.
