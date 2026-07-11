import numpy as np
import cv2
img = cv2.imread(r"C:\Users\Piyush\Documents\cv images\choice-variation-concept.webp")
if img is None:
    print("Error: Image not found.")
    exit()
# resize 
img = cv2.resize(img, (600,600), interpolation=cv2.INTER_AREA)
#rectangle drawing on the image 
cv2.rectangle(img, (50, 50), (200, 200), (0,255,0), 3)

#circle on the image
cv2.circle(img, (300, 300), 50, (255,0,0), -1)

#line on the image
cv2.line(img, (400, 400), (500, 500), (0,0,255), 2)
cv2.line(img, (400, 420), (500, 520), (0,0,255), 2)
cv2.line(img, (400, 440), (500, 540), (0,0,255), 2)

#adding watermark to the image
overlay = img.copy()
height,width = img.shape[:2]
cv2.rectangle(overlay, (0, height-80), (width, height), (0, 0, 0), -1)

#blending the overlay with the original image
alpha = 0.7 
beta = 0.3 
watermark_img = cv2.addWeighted(img, alpha , overlay, beta , 0)

#text in the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(watermark_img, "Hey there! Piyush this side", (50, height-30), font, 1.2 , (255, 255, 255), 2)

cv2.imshow('Image with Shapes and Watermark', watermark_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
