import cv2
import numpy as np
import time
# made a class to encapsulate all the functionality of our background remover app
class BackgroundRemover:
    def __init__(self, camera_id=0, bg_image_path=r"C:\Users\Piyush\Documents\cv_journey\Week3\face_img\bg.webp"):
        """This runs automatically when we create our object."""
        self.cap = cv2.VideoCapture(camera_id)
        self.bg_image = cv2.imread(bg_image_path)
        self.mode = 1  # 1: Image, 2: Blur, 3: Normal
        self.setup_trackbars()
        
    def empty(self, a):
        pass

    def setup_trackbars(self):
        """Creates the settings window for tuning our mask."""
        cv2.namedWindow("Settings")
        cv2.resizeWindow("Settings", 640, 250)
        cv2.createTrackbar("Hue Min", "Settings", 40, 179, self.empty)
        cv2.createTrackbar("Hue Max", "Settings", 90, 179, self.empty)
        cv2.createTrackbar("Sat Min", "Settings", 40, 255, self.empty)
        cv2.createTrackbar("Sat Max", "Settings", 255, 255, self.empty)
        cv2.createTrackbar("Val Min", "Settings", 40, 255, self.empty)
        cv2.createTrackbar("Val Max", "Settings", 255, 255, self.empty)

    def get_alpha_mask(self, frame):
        """Handles HSV, Morphology, Erosion, and Blurring."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Read sliders
        h_min = cv2.getTrackbarPos("Hue Min", "Settings")
        h_max = cv2.getTrackbarPos("Hue Max", "Settings")
        s_min = cv2.getTrackbarPos("Sat Min", "Settings")
        s_max = cv2.getTrackbarPos("Sat Max", "Settings")
        v_min = cv2.getTrackbarPos("Val Min", "Settings")
        v_max = cv2.getTrackbarPos("Val Max", "Settings")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        # 1. Color Mask
        bg_mask = cv2.inRange(hsv, lower, upper)
        
        # 2. Smooth (Morphology)
        kernel = np.ones((5, 5), np.uint8)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Invert (You = White)
        fg_mask = cv2.bitwise_not(bg_mask)
        
        # 4. Refine Edges (Erosion + Blur)
        erode_kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, erode_kernel, iterations=2)
        fg_mask = cv2.GaussianBlur(fg_mask, (15, 15), 0)
        
        # 5. Convert to 0.0 - 1.0 Alpha scale
        alpha = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        return alpha

    def apply_background(self, frame, alpha):
        """Performs the Alpha Blending math with the chosen background."""
        foreground = frame.astype(float)
        
        if self.mode == 1 and self.bg_image is not None:
            bg_resized = cv2.resize(self.bg_image, (frame.shape[1], frame.shape[0]))
            background = bg_resized.astype(float)
            
        elif self.mode == 2:
            background = cv2.GaussianBlur(frame, (51, 51), 0).astype(float)
            
        else:
            background = foreground.copy()
            
        # The Magic Blend Equation
        blended = cv2.multiply(alpha, foreground) + cv2.multiply(1.0 - alpha, background)
        return blended.astype(np.uint8)

    def run(self):
        """The main loop that runs our pipeline."""
        print("Zoom Clone App Initialized!")
        print("Press '1' for Image | '2' for Blur | '3' for Normal | 'q' to Quit")
        
        prev_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            
            # cropped the frame
            h, w = frame.shape[:2]

            crop_w, crop_h = 350,250 
            
            
            if w >= crop_w and h >= crop_h:
                start_x = (w - crop_w) // 2
                start_y = (h - crop_h) // 2
                frame = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            
            alpha_mask = self.get_alpha_mask(frame)
            final_output = self.apply_background(frame, alpha_mask)
            
            # Add FPS counter
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))
            prev_time = curr_time
            cv2.putText(final_output, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Week 3: Pro Zoom Clone", final_output)
            
            # Listen for keyboard presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'): self.mode = 1
            elif key == ord('2'): self.mode = 2
            elif key == ord('3'): self.mode = 3
            elif key == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()


# 
if __name__ == "__main__":
    # 1. Create the App
    my_app = BackgroundRemover(camera_id=0, bg_image_path=r"C:\Users\Piyush\Documents\cv_journey\Week3\face_img\bg.webp")
    
    # 2. Run the App
    my_app.run()