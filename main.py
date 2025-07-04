import json
import signal
import subprocess
import sys
import threading
import time

import cv2

# Configuration

# Load the pre-trained face and eye cascade classifiers from OpenCV's data directory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")  # Better eye model

# Check if the classifiers loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load cascade classifiers. Check OpenCV installation.")
    exit()

# Load communication data
try:
    with open('data.json', 'r', encoding='utf-8') as f:
        communication_data = json.load(f)
    categories = communication_data['categories']
    print(f"Loaded {len(categories)} categories for communication")
except FileNotFoundError:
    print("Error: data.json file not found.")
    exit()
except json.JSONDecodeError:
    print("Error: Invalid JSON in data.json.")
    exit()

# Audio feedback function
def play_beep():
    """Play beep sound for selection confirmation"""
    try:
        subprocess.run(['afplay', 'beep.mp3'], check=False)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Could not play beep sound: {e}")

# Navigation state variables
navigation_mode = "categories"  # "categories" or "actions"
current_category_index = 0
current_action_index = 0
selected_category = None
blink_start_time = None
is_long_blink = False
LONG_BLINK_THRESHOLD = 0.8  # 0.8 seconds to select (more responsive)
selection_feedback_time = 0
FEEDBACK_DURATION = 2.0  # Show selection feedback for 2 seconds

# Start video capture (0 is usually your default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Performance optimization: set lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Print startup instructions
print("=" * 60)
print("EYE BLINK COMMUNICATION SYSTEM")
print("=" * 60)
print("HOW TO USE:")
print("• SHORT BLINK: Navigate through options")
print("• LONG BLINK (hold for 0.8 seconds): Select current option")
print("• The system will beep IMMEDIATELY when selection is made")
print("• You can open your eyes after hearing the beep")
print("• Press 'q' to quit, 'b' to go back, 'r' to reset")
print("=" * 60)
print("Starting camera...")
print()

# Variables for FPS calculation
prev_time = time.time()
fps = 0

# Variables for blink detection
last_blink_time = time.time()
blink_cooldown = 0.5  # Cooldown between navigation blinks
blink_counter = 0

# Blink detection state variables
eye_closed_frames = 0
eye_open_frames = 0
eye_state = "open"  # Can be "open", "closed", or "transition"
EYE_CLOSED_THRESHOLD = 3  # Frames to confirm eyes are closed
EYE_OPEN_THRESHOLD = 3  # Frames to confirm eyes are open

# Eye aspect ratio (EAR) tracking for improved detection
prev_eyes_count = 0
confidence_level = 0  # For tracking detection confidence
MAX_CONFIDENCE = 3

def handle_navigation():
    """Handle short blink - navigate through options"""
    global current_category_index, current_action_index, navigation_mode, last_blink_time
    
    nav_time = time.time()
    if nav_time - last_blink_time < blink_cooldown:
        return
    
    last_blink_time = nav_time
    
    if navigation_mode == "categories":
        current_category_index = (current_category_index + 1) % len(categories)
        print(f"→ Navigating to category: {categories[current_category_index]['name']}")
    elif navigation_mode == "actions":
        actions = selected_category['actions']
        current_action_index = (current_action_index + 1) % len(actions)
        print(f"→ Navigating to action: {actions[current_action_index]['description']}")

def handle_selection():
    """Handle long blink - make a selection"""
    global navigation_mode, selected_category, current_action_index, selection_feedback_time
    global current_category_index
    
    current_time = time.time()
    selection_feedback_time = current_time
    
    # Play beep immediately to signal selection
    threading.Thread(target=play_beep, daemon=True).start()
    
    if navigation_mode == "categories":
        selected_category = categories[current_category_index]
        navigation_mode = "actions"
        current_action_index = 0
        print(f"✓ SELECTED CATEGORY: {selected_category['name']}")
        print("Now selecting action - blink to navigate, long blink to select")
    elif navigation_mode == "actions":
        selected_action = selected_category['actions'][current_action_index]
        print("=" * 60)
        print(f"✓ FINAL SELECTION: {selected_action['description']}")
        print(f"  Category: {selected_category['name']}")
        print("=" * 60)
        # Reset to categories for next communication
        navigation_mode = "categories"
        selected_category = None
        current_category_index = 0

def draw_navigation_ui(frame):
    """Draw the navigation interface on the frame"""
    # Background for UI
    cv2.rectangle(frame, (10, 150), (630, 470), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 150), (630, 470), (255, 255, 255), 2)
    
    y_offset = 180
    line_height = 25
    
    if navigation_mode == "categories":
        cv2.putText(frame, "SELECT CATEGORY (Long blink to select, short to navigate)", 
                   (15, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show categories with current selection highlighted
        for i, category in enumerate(categories):
            color = (0, 255, 0) if i == current_category_index else (255, 255, 255)
            text = f"{'→ ' if i == current_category_index else '  '}{category['name']}"
            cv2.putText(frame, text, (15, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == current_category_index else 1)
    
    elif navigation_mode == "actions" and selected_category:
        cv2.putText(frame, f"CATEGORY: {selected_category['name']}", 
                   (15, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "SELECT ACTION (Long blink to select, short to navigate)", 
                   (15, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        actions = selected_category['actions']
        # Show up to 10 actions at a time for better visibility
        start_idx = max(0, current_action_index - 5)
        end_idx = min(len(actions), start_idx + 10)
        
        for i in range(start_idx, end_idx):
            action = actions[i]
            color = (0, 255, 0) if i == current_action_index else (255, 255, 255)
            text = f"{'→ ' if i == current_action_index else '  '}{action['description']}"
            cv2.putText(frame, text, (15, y_offset + 40 + (i - start_idx) * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if i == current_action_index else 1)
    
    # Show blink progress for long blinks
    if blink_start_time is not None:
        blink_progress = min(1.0, (time.time() - blink_start_time) / LONG_BLINK_THRESHOLD)
        progress_width = int(300 * blink_progress)
        # Background bar
        cv2.rectangle(frame, (15, 460), (315, 480), (50, 50, 50), -1)
        # Progress bar
        if blink_progress >= 1.0:
            # Progress reached 100% - show full green bar
            cv2.rectangle(frame, (15, 460), (315, 480), (0, 255, 0), -1)
            if is_long_blink:
                cv2.putText(frame, "SELECTED! ✓", (320, 475), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "SELECTING... 100%", (320, 475), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Progress - yellow/orange bar
            cv2.rectangle(frame, (15, 460), (15 + progress_width, 480), (0, 165, 255), -1)
            cv2.putText(frame, f"Hold blink... {blink_progress:.0%}", (320, 475), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show recent selection feedback
    if time.time() - selection_feedback_time < FEEDBACK_DURATION:
        cv2.putText(frame, "SELECTION MADE! ✓", (400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Additional UI improvements for better visibility
def draw_instructions(frame):
    """Draw basic instructions on the frame"""
    cv2.putText(frame, "Short blink: Navigate | Long blink (0.8s): Select", 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

    # State machine for blink detection with navigation
    if len(faces) > 0:  # Only process if a face is detected
        if eyes_count >= 1:  # Eyes are open
            eye_open_frames += 1
            eye_closed_frames = 0
            
            # Check if eyes just opened (transition from closed to open)
            if eye_state == "closed" and eye_open_frames >= EYE_OPEN_THRESHOLD:
                # Eyes have opened - check if we had a blink and it wasn't already processed
                if blink_start_time is not None and not is_long_blink:
                    blink_duration = current_time - blink_start_time
                    if blink_duration >= LONG_BLINK_THRESHOLD:
                        # Long blink detected - make selection
                        print(f"Long blink completed! Duration: {blink_duration:.2f}s")
                        handle_selection()
                        is_long_blink = True
                    elif blink_duration < LONG_BLINK_THRESHOLD:
                        # Short blink detected - navigate
                        print(f"Short blink detected! Duration: {blink_duration:.2f}s")
                        handle_navigation()
                
                # Reset blink tracking after processing
                if not is_long_blink or eye_open_frames >= EYE_OPEN_THRESHOLD + 2:
                    blink_start_time = None
                    is_long_blink = False
                
                eye_state = "open"
                
        else:  # Eyes are closed (no eyes detected)
            eye_closed_frames += 1
            eye_open_frames = 0
            
            # Check if eyes just closed (transition from open to closed)
            if eye_state == "open" and eye_closed_frames >= EYE_CLOSED_THRESHOLD:
                # Start tracking blink duration
                blink_start_time = current_time
                eye_state = "closed"
                is_long_blink = False
                print("Eyes closed - tracking blink duration...")
            
            # Check for long blink while eyes are still closed
            elif eye_state == "closed" and blink_start_time is not None and not is_long_blink:
                blink_duration = current_time - blink_start_time
                if blink_duration >= LONG_BLINK_THRESHOLD:
                    # Long blink threshold reached - make selection immediately
                    print(f"Long blink threshold reached! Duration: {blink_duration:.2f}s")
                    handle_selection()
                    is_long_blink = True
                    # Keep blink_start_time for a moment to show 100% progress
                    # It will be reset when eyes open
                    
    else:
        # No face detected - reset counters
        eye_open_frames = 0
        eye_closed_frames = 0
        if blink_start_time is not None:
            print("Face lost - resetting blink tracking")
        blink_start_time = None
        is_long_blink = False
        eye_state = "open"

    # Draw navigation interface
    draw_navigation_ui(frame)
    draw_instructions(frame)
    
    # Display eye state and system info
    eye_status = f"Eyes: {('Open' if eyes_detected else 'Closed') if len(faces) > 0 else 'No Face'} | State: {eye_state}"
    cv2.putText(frame, eye_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display detection stats
    cv2.putText(frame, f"FPS: {fps:.1f} | Eyes detected: {eyes_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Navigation: {navigation_mode.title()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Debug blink tracking
    if blink_start_time is not None:
        blink_duration = current_time - blink_start_time
        blink_progress = min(1.0, blink_duration / LONG_BLINK_THRESHOLD)
        cv2.putText(frame, f"Blink: {blink_duration:.2f}s ({blink_progress:.0%})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # Show the video feed with instructions
    cv2.imshow("Eye Blink Communication System", frame)
    
    # Handle keyboard input for testing and navigation
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to quit
        break
    elif key == ord("b"):  # Press 'b' to go back (for testing)
        if navigation_mode == "actions":
            navigation_mode = "categories"
            selected_category = None
            current_category_index = 0
            print("Returned to categories")
    elif key == ord("r"):  # Press 'r' to reset
        navigation_mode = "categories"
        selected_category = None
        current_category_index = 0
        current_action_index = 0
        print("Reset to beginning")

def safe_cleanup():
    """Safely clean up resources"""
    try:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete.")
    except Exception as e:
        print(f"Cleanup error: {e}")

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    safe_cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

safe_cleanup()
