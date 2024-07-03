import cv2
import pytesseract

# Path to Tesseract executable (change this based on your system configuration)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load the Haar Cascade for license plate detection
cascade_path = 'haarcascade_russian_plate_number.xml'
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Function to perform OCR on detected license plates
def ocr_plate(img, plate_coords):
    x, y, w, h = plate_coords
    plate_img = img[y:y+h, x:x+w]
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_plate, config='--psm 8 --oem 3')
    return text.strip()

# Initialize camera
cap = cv2.VideoCapture(0)  # Adjust camera index if necessary

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform license plate detection
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    # Draw rectangles around detected plates and perform OCR
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plate_text = ocr_plate(frame, (x, y, w, h))
        cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('License Plate Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
