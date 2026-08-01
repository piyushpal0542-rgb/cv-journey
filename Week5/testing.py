import tensorflow as tf
import numpy as np
import cv2
import os

def test_image(image_path, model_path="augmented_cats_vs_dogs_model.keras"):
    print("Waking up the AI Brain...")
    
    # 1. Load the saved model 
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Did you finish Thursday's training?")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # 2. Read the image with OpenCV for displaying later
    display_img = cv2.imread(image_path)
    if display_img is None:
        print(f"Error: Could not load image at '{image_path}'.")
        return

    print(f"Processing '{image_path}'...")
    
    # 3. Preprocess the image EXACTLY how the AI was trained
    # We must shrink it to 150x150
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    
    # Normalize pixels (scale 0-255 down to 0.0-1.0)
    img_array = img_array / 255.0
    
    # Keras expects a "batch" of images, not just one. 
    # expand_dims turns our 1 image into a batch of 1. Shape becomes (1, 150, 150, 3)
    img_array = tf.expand_dims(img_array, 0)
    
    # 4. Make the Prediction!
    print("Analyzing image...")
    prediction = model.predict(img_array)
    
    # 5. Decode the output
    # Since we used a 'sigmoid' activation, the output is a single percentage between 0 and 1
    # 0 = Cat, 1 = Dog
    score = prediction[0][0]
    
    if score > 0.5:
        label = "DOG"
        confidence = score * 100
        color = (0, 255, 0) # Green for Dog
    else:
        label = "CAT"
        confidence = (1.0 - score) * 100
        color = (255, 0, 0) # Blue for Cat
        
    result_text = f"AI Says: {label} ({confidence:.1f}% sure)"
    print(result_text)
    
    # 6. Display the result visually
    # Resize the display image so it fits nicely on screen
    h, w = display_img.shape[:2]
    if w > 800 or h > 800:
        scale = 800 / max(w, h)
        display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))
        
    cv2.putText(display_img, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Week 5: AI Prediction", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    my_test_image = r"C:\Users\Piyush\Documents\cv images\test.webp" 
    
    test_image(my_test_image)