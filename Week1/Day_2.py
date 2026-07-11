import numpy as np 
import cv2
def process_img(img_path):
    img = cv2.imread(img_path)
    if img is None :
        print("Error: Image not found.")
        return
    height, width = img.shape[:2]
    # resize the image (500x500)
    new_dimension = (500,500)
    img_resized = cv2.resize(img, new_dimension, interpolation=cv2.INTER_AREA)

    #cropping of the image from the centre
    start_x, end_x  = int(height*0.25) , int(height*0.75)
    start_y, end_y  = int(width*0.25) , int(width*0.75)
    img_cropped = img[start_y:end_y , start_x:end_x]

    #rotating the image by 45 degrees anti clock-wise
    center = (width // 2, height // 2)
    angle = 45
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

    #printing the images
    cv2.imshow("Original Image", img)
    cv2.imshow("Resized Image", img_resized)
    cv2.imshow("Cropped Image", img_cropped)
    cv2.imshow("Rotated Image", img_rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_img(r"C:\Users\Piyush\Documents\cv images\choice-variation-concept.webp")