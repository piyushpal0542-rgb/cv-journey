import cv2

def detect_faces(image_path):
    # Load the pre-trained Haar Cascade model for frontal faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image and convert it to grayscale 
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not find image. Check your path!")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"I found {len(faces)} face(s) in this image!")

    # Draw a blue rectangle around every face it found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        # Add a little text label above the box
        cv2.putText(img, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the result
    cv2.imshow("Week 3: Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this to the name of an image saved in your Week 3 folder!
    my_image = r"C:\Users\Piyush\Documents\cv_journey\Week3\face_img\group.webp"
    detect_faces(my_image)