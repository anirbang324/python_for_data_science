import cv2
from fer import FER

# Initialize camera and emotion detector
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_detector = FER()

while True:
    ret, frame = cam.read()

    if not ret:
        continue

    # Detect faces using Haar Cascade
    allfaces = detector.detectMultiScale(frame, 1.5, 3)

    for (x, y, w, h) in allfaces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region from the frame
        face_region = frame[y:y+h, x:x+w]

        # Predict emotions using the FER library
        emotion_results = emotion_detector.detect_emotions(face_region)

        # Check if any emotions were detected
        if emotion_results:
            # Get the top emotion from the detection
            top_emotion = emotion_results[0]['emotions']
            emotion_label = max(top_emotion, key=top_emotion.get)

            # Display the emotion label on the screen
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show the updated frame with emotion detection
    cv2.imshow("Emotion Detection", frame)

    # Check if the 'q' key is pressed to quit
    keyprsd = cv2.waitKey(1) & 0xFF
    if keyprsd == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
