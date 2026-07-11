import cv2
import numpy as np

def preprocess_frame(frame):
    """Resizes, Grayscales, Blurs, and finds Canny edges for a single video frame."""
    target_height = 800.0
    h, w = frame.shape[:2]
    scale = target_height / h
    frame_resized = cv2.resize(frame, (int(w * scale), int(target_height)))

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # LOWERED the thresholds to handle blurry/low-light webcam feeds better
    edges = cv2.Canny(blurred, 50, 150)

    return frame_resized, edges

def find_document_contour(edges):
    """Finds the 4-corner document contour without freezing the video."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        
        if len(approx) == 4 and cv2.contourArea(contour) > 30000:
            return approx
            
    # NEW FALLBACK: If hands/thumbs are breaking the straight edges, 
    # force a perfect 4-corner rectangle around the biggest shape!
    if len(contours) > 0:
        biggest_contour = contours[0]
        if cv2.contourArea(biggest_contour) > 30000:
            rect = cv2.minAreaRect(biggest_contour)
            box = cv2.boxPoints(rect)
            return np.int32(box)
            
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_document(img, contour):
    pts = contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

if __name__ == "__main__":
    # 1. Initialize the IP Webcam 
    ip_camera_url = "http://192.168.1.2:8080/video" 
    
    cap = cv2.VideoCapture(ip_camera_url)
    
    #This eliminates Wi-Fi delay!
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Webcam started. Press 'q' to quit.")

    # 2. Start the infinite video loop
    while True:
        # Read a single frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Process the frame
        resized_frame, edges = preprocess_frame(frame)
        doc_contour = find_document_contour(edges)
        
        # If we find a document, draw the box and warp it
        if doc_contour is not None:
            # We draw on a COPY so the green lines don't get warped into our final scan
            display_frame = resized_frame.copy()
            cv2.drawContours(display_frame, [doc_contour], -1, (0, 255, 0), 3)
            
            scanned_doc = warp_document(resized_frame, doc_contour)
            
            # Show the warped result in a separate window
            cv2.imshow("Scanned Result", scanned_doc)
        else:
            display_frame = resized_frame
            
        # Show the live webcam feed
        cv2.imshow("Live Scanner", display_frame)
        
        # 3. Listen for the 'q' key to quit the loop (runs every 1 millisecond)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()