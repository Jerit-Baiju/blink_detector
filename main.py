import time

import cv2

# Load the pre-trained face and eye cascade classifiers from OpenCV's data directory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")  # Better eye model

# Check if the classifiers loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load cascade classifiers. Check OpenCV installation.")
    exit()

# Start video capture (0 is usually your default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Performance optimization: set lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for FPS calculation
prev_time = time.time()
fps = 0

# Variables for blink detection
last_blink_time = time.time()
blink_cooldown = 0.3  # Reduced cooldown to catch more rapid blinks
blink_counter = 0

# Blink detection state variables
eye_closed_frames = 0
eye_open_frames = 0
eye_state = "open"  # Can be "open", "closed", or "transition"
EYE_CLOSED_THRESHOLD = 2  # Reduced threshold to detect faster blinks
EYE_OPEN_THRESHOLD = 1  # Reduced threshold to reset faster

# Eye aspect ratio (EAR) tracking for improved detection
prev_eyes_count = 0
confidence_level = 0  # For tracking detection confidence
MAX_CONFIDENCE = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Calculate FPS
    current_time = time.time()
    elapsed = current_time - prev_time
    if elapsed > 0:
        fps = 1 / elapsed
    prev_time = current_time

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast in varying light conditions
    gray = cv2.equalizeHist(gray)

    # Improve performance by reducing search area and tuning parameters
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    eyes_detected = False
    eyes_count = 0

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest for eyes - focus on upper part of face
        roi_gray = gray[y : y + int(h * 0.6), x : x + w]  # Focus on upper 60% of face
        roi_color = frame[y : y + int(h * 0.6), x : x + w]

        # Detect eyes in the face region with adjusted parameters
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(25, 25), maxSize=(80, 80))

        eyes_count = len(eyes)
        for ex, ey, ew, eh in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eyes_detected = True

    # State machine for blink detection
    if len(faces) > 0:  # Only process if a face is detected
        if eyes_count >= 1:  # Eyes are open
            eye_open_frames += 1
            eye_closed_frames = 0
            if eye_state == "closed" and eye_open_frames >= EYE_OPEN_THRESHOLD:
                eye_state = "open"
        else:  # Eyes are closed
            eye_closed_frames += 1
            eye_open_frames = 0
            if eye_state == "open" and eye_closed_frames >= EYE_CLOSED_THRESHOLD:
                # This is a blink
                current_time = time.time()
                if current_time - last_blink_time > blink_cooldown:
                    blink_counter += 1
                    print(f"Blink detected! Count: {blink_counter}")
                    last_blink_time = current_time
                eye_state = "closed"
    else:
        # No face detected - reset counters
        eye_open_frames = 0
        eye_closed_frames = 0

    # Display eye state
    eye_status = "Eyes: " + ("Open" if eyes_detected else "Closed" if len(faces) > 0 else "No Face")
    cv2.putText(frame, eye_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display detection stats
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Eye Blink Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()
