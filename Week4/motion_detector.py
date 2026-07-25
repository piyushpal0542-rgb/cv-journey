import cv2
import csv
import os
import time
from datetime import datetime

if __name__ == "__main__":
    # Setup directories and CSV logging
    os.makedirs("captured_frames", exist_ok=True)
    
    csv_file = open("security_log.csv", mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Event", "Image_File"])
    
    cap = cv2.VideoCapture(0)
    reference_frame = None
    last_capture_time = 0  # Cooldown timer

    print("Security Camera Active! Press 'r' to reset background, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        frame = cv2.flip(frame, 1)
        
        # Crop frame to standard YouTube ratio
        h, w = frame.shape[:2]
        crop_w, crop_h = 640, 480
        if w >= crop_w and h >= crop_h:
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            frame = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]

        # Preprocessing: Grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if reference_frame is None:
            reference_frame = gray
            continue 

        # Motion Detection
        frame_diff = cv2.absdiff(reference_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Noise Filtering (Dilate to merge blobs)
        thresh = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False

        for contour in contours:
            # Ignore small movements/shadows
            if cv2.contourArea(contour) < 5000:
                continue
                
            motion_detected = True 
            
            # Draw bounding box
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "MOTION", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Logging & Saving Frames (with 5-second cooldown)
        if motion_detected:
            current_time = time.time()
            
            if current_time - last_capture_time > 5.0:
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_filename = f"captured_frames/intruder_{timestamp_str}.jpg"
                
                cv2.imwrite(image_filename, frame)
                
                csv_writer.writerow([timestamp_str, "Motion Detected", image_filename])
                csv_file.flush() 
                
                print(f"Captured! Logged {image_filename} to CSV.")
                last_capture_time = current_time

        cv2.imshow("Security Camera App", frame)
        cv2.imshow("Threshold view", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("Background reset!")
            reference_frame = gray
        elif key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()