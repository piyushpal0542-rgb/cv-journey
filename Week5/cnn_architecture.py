import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import os

if __name__ == "__main__":
    print("Loading Local Datasets...")

    train_dir = os.path.join("dataset", "cats_and_dogs_filtered", "train")
    val_dir = os.path.join("dataset", "cats_and_dogs_filtered", "validation")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(150, 150), batch_size=16, label_mode='binary'
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(150, 150), batch_size=16, label_mode='binary'
    )
    
    # Normalize pixels (scale 0-255 down to 0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    print("Building Data Augmentation Pipeline...")

    
    # This slightly alters the images every single time the AI sees them.
    data_augmentation = Sequential([
        RandomFlip("horizontal", input_shape=(150, 150, 3)), # Flips image left/right
        RandomRotation(0.2), # Rotates the image up to 20%
        RandomZoom(0.2),     # Zooms in or out up to 20%
    ])

    print("Building the Augmented CNN Architecture...")

    # Build the CNN, but plug the Augmentation in as the VERY FIRST layer!
    model = Sequential([
        # LAYER 0: The Augmentation Block
        data_augmentation, 

        # The rest of our architecture from Wednesday remains exactly the same
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    print("Starting Retraining with Augmented Data! Open Task Manager...")
    
    # Retrain the model. We increase epochs to 15 because the AI has harder data to learn now!
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=15 
    )

    # Save this
    model.save("augmented_cats_vs_dogs_model.keras")
    print("Training Complete! Smarter brain saved as 'augmented_cats_vs_dogs_model.keras'")