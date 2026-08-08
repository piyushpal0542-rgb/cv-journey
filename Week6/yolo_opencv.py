import cv2
import numpy as np
import time
import os

# The 80 COCO Classes
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
           "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

if __name__ == "__main__":
    print("Loading YOLOv8 ONNX Brain...")
    
    onnx_path = "yolov8n.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: Could not find {onnx_path}.")
        exit()
        
    net = cv2.dnn.readNetFromONNX(onnx_path)
    cap = cv2.VideoCapture(0)
    
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        original_height, original_width = frame.shape[:2]

        # Preprocess & Forward Pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        # Decoding
        boxes, confidences, class_ids = [], [], []
        predictions = np.squeeze(outputs).T 
        x_scale, y_scale = original_width / 640.0, original_height / 640.0

        for row in predictions:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence > 0.4:
                cx, cy, w, h = row[0:4]
                left = int((cx - w / 2) * x_scale)
                top = int((cy - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)

                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # NMS Filter
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

        # Create an empty dictionary to hold the counts for THIS specific frame
        current_counts = {}

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = CLASSES[class_ids[i]]
                label = f"{class_name}: {confidences[i]*100:.1f}%"
                
                # Increment the count for this object in our dictionary
                current_counts[class_name] = current_counts.get(class_name, 0) + 1
                
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- DRAW THE LIVE COUNTS ON SCREEN ---
        # We will draw a neat little scoreboard in the top right corner
        y_offset = 30
        cv2.putText(frame, "LIVE COUNT:", (original_width - 250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        for obj, count in current_counts.items():
            y_offset += 30
            text = f"{obj.capitalize()}: {count}"
            cv2.putText(frame, text, (original_width - 250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Calculate and Draw FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Week 6: Object Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()