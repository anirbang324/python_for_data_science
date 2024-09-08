import cv2
from deepface import DeepFace

# Initialize camera and face detector
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0  # To track frame count

while True:
    ret, frame = cam.read()

    if not ret:
        continue

    frame_count += 1

    # Resize the frame to speed up detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Only analyze every 10th frame to reduce load
    if frame_count % 10 == 0:
        # Detect faces using Haar Cascade
        allfaces = detector.detectMultiScale(small_frame, 1.5, 3)

        for (x, y, w, h) in allfaces:
            # Rescale face coordinates back to the original frame size
            x, y, w, h = x * 2, y * 2, w * 2, h * 2

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face region from the frame
            face_region = frame[y:y+h, x:x+w]

            try:
                # Predict age using DeepFace
                age_results = DeepFace.analyze(face_region, actions=['age'], enforce_detection=False)

                # If more than one face is detected, DeepFace returns a list
                if isinstance(age_results, list):
                    age = age_results[0]['age']  # Get the age for the first face
                else:
                    age = age_results['age']  # Direct access if only one face

                # Display the age on the screen
                cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            except Exception as e:
                print(f"Error in detecting age: {str(e)}")

    # Show the updated frame with age detection
    cv2.imshow("Age Detection", frame)

    # Check if the 'q' key is pressed to quit
    keyprsd = cv2.waitKey(1) & 0xFF
    if keyprsd == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
