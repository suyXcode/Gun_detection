import numpy as np
import cv2
import imutils
import datetime

# Load the gun cascade
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Open the camera
camera = cv2.VideoCapture(0)

firstFrame = None

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        break
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns
    gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
    
    gun_exist = len(gun) > 0  # Check if any gun is detected

    # Draw rectangles around detected guns
    for (x, y, w, h) in gun:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if firstFrame is None:
        firstFrame = gray
        continue

    # Add timestamp to the frame
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display detection status
    if gun_exist:
        print("Guns detected")
        cv2.putText(frame, "Gun Detected!", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the video feed with detections
    cv2.imshow("Security Feed", frame)

    # Press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
