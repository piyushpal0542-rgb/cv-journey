import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}.")
        return None, None
    
    # Resize
    target_height = 800.0
    h, w = img.shape[:2]
    scale = target_height / h
    img_resized = cv2.resize(img, (int(w * scale), int(target_height)))

    # Grayscale & Blur
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 75, 200)

    return img_resized, edges


def find_document_contour(edges, img_resized):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    document_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        # 5% smoothing for real-world corners
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        
        # Must have 4 corners AND be larger than 30,000 pixels
        if len(approx) == 4 and cv2.contourArea(contour) > 30000:
            document_contour = approx
            break

    if document_contour is not None:
        # Draw the target box for visual feedback
        result_img = img_resized.copy()
        cv2.drawContours(result_img, [document_contour], -1, (0, 255, 0), 3)
        cv2.imshow("Document Target", result_img)
        return document_contour
    else:
        print("Could not find a document contour.")
        return None

# PERSPECTIVE TRANSFORM FUNCTION

def order_points(pts):
    #Sorts the 4 corners into: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right point will have the smallest difference,the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def warp_document(img, contour):
    # Flattens the angled document into a top-down view
    # 1. Reformat the contour points and order them mathematically
    pts = contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 2. Calculate the maximum width of the new flat image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 3. Calculate the maximum height of the new flat image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 4. Define the dimensions of our new flat destination image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 5. Calculate the Perspective Transform Matrix and apply it!
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    return warped

if __name__ == "__main__":
    # Test it with the perfect image you just generated!
    image_path = r"C:\Users\Piyush\Documents\cv images\document.png"
    
    resized_img, canny_edges = preprocess_image(image_path)
    
    if resized_img is not None:
        doc_contour = find_document_contour(canny_edges, resized_img)
        
        # Run the new warp function if we successfully found the document
        if doc_contour is not None:
            scanned_doc = warp_document(resized_img, doc_contour)
            
            # Show the final flattened result!
            cv2.imshow("Final Scanned Document", scanned_doc)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




