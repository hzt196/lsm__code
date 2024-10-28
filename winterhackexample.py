import cv2
import numpy as np

# Open the webcam (0 is the default webcam index)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break

    # Convert the image from BGR color space to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for detecting the red laser point
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])

    # Create a mask that identifies the red areas in the frame
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise AND mask and original image to only show the red areas
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours of the detected red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for cnt in contours:
        # Get the minimum enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))  # Center of the detected laser point
        
        # Filter out small detections by checking the radius
        if radius > 5:
            # Draw a circle around the detected laser point on the original frame
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            # Display the coordinates of the laser point
            cv2.putText(frame, f"Laser Point: {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the original frame with the laser point marked
    cv2.imshow("Laser Detection", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
