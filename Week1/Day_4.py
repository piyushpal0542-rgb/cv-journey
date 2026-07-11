import numpy as np
import cv2
#empty function 
def empty(a):
    pass
#creating windows
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
# trackbars 
cv2.createTrackbar("HUE Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("HUE Max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("SAT Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("SAT Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("VAL Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("VAL Max", "Trackbars", 255, 255, empty)

img = cv2.imread(r"C:\Users\Piyush\Documents\cv images\Gemini_Generated_Image_75x35l75x35l75x3.png")
if img is None:
    print("Error: Could not read the image.")
    exit()
#resize the image 
img = cv2.resize(img, (500,500))
#while loop to continuously get the trackbar values and apply the mask
while True:
    #converting bgr to hsv
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #set the trackbar values 
    h_min = cv2.getTrackbarPos("HUE Min", "Trackbars")
    h_max = cv2.getTrackbarPos("HUE Max", "Trackbars")
    s_min = cv2.getTrackbarPos("SAT Min", "Trackbars")
    s_max = cv2.getTrackbarPos("SAT Max", "Trackbars")
    v_min = cv2.getTrackbarPos("VAL Min", "Trackbars")
    v_max = cv2.getTrackbarPos("VAL Max", "Trackbars")
# create a numpy array for storing the max and min values

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
#creating a mask 
    mask = cv2.inRange(HSV_img, lower, upper)
#applying the mask adding it to original image
    result = cv2.bitwise_and(img, img, mask= mask)
# applying the result 
    cv2.imshow("Original Image", img)    
    cv2.imshow("HSV Image ", HSV_img)
    cv2.imshow("Mask image", mask)
    cv2.imshow("Result Image", result)
# make a if statemen to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()