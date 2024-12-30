import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #for floating point representation issue with pc

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import json


# Definition of model
def create_model():
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
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#Training model
def train_model():
    BATCH_SIZE=32
    DATASET_PATH = 'data/melanoma'
    TRAIN_DIR=os.path.join(DATASET_PATH, 'train')
    TEST_DIR=os.path.join(DATASET_PATH, 'test')

    # Data augmentation
    train_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    test_datagen=ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator=test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = test_generator.samples // BATCH_SIZE 

    # Print dataset information
    print(f"Total training images: {train_generator.samples}")
    print(f"Total test images: {test_generator.samples}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Class indices: {train_generator.class_indices}")

    #Create and train model
    model=create_model()
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    #Training model to explicitly capture history
    try:
        history=model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=test_generator,
            validation_steps=validation_steps,
            verbose=1,
            callbacks=[early_stopping]
        )   
    
        # Saving of model
        # model.save(model, 'models/melanoma_model.h5', save_format='.h5')
        model.save('models/melanoma_model.h5')
        print("\nTraining completed and model saved!")
        # model_json = model.to_json()
        # with open('models/model_architecture.json', 'w') as json_file:
        #     json_file.write(model_json)
        # model.save_weights('models/model_weights.weights.h5')
        
        
        # Print final model metrics
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        final_loss = history.history['loss'][-1]
        final_val_loss = model.history.history['val_loss'][-1]
        
        print(f"\nFinal training accuracy: {final_accuracy:.4f}")
        print(f"\nFinal vaidation accuracy: {final_val_accuracy:.4f}")
        print(f"\nFinal training loss: {final_loss:.4f}")
        print(f"\nFinal vaidation loss: {final_val_loss:.4f}")
            
    except Exception as e:
        print(f"An error has occured: {str(e)}")
        
if __name__ == '__main__':
    train_model()
        



