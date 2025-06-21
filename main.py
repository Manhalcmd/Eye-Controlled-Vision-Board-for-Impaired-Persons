import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import pyttsx3
import threading
import time

# Initialize TTS engine
engine = pyttsx3.init()

# Load sounds with error handling
try:
    sound = pyglet.media.load("sound.wav", streaming=False)
    left_sound = pyglet.media.load("left.wav", streaming=False)
    right_sound = pyglet.media.load("right.wav", streaming=False)
    siren_sound = pyglet.media.load("siren.wav", streaming=False)
except:
    print("Warning: Some sound files not found. Audio feedback will be limited.")
    sound = left_sound = right_sound = siren_sound = None

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Create board for text display
board = np.zeros((450, 1400), np.uint8)
board[:] = 255

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    print("Error: shape_predictor_68_face_landmarks.dat not found. Please download it.")
    exit()

# Create image grid display
image_grid = np.zeros((900, 900, 3), np.uint8)

# Image sets
keys_set_1 = {
    0: "image1", 1: "image2", 2: "image3", 
    3: "image4", 4: "image5", 5: "image6", 
    6: "image7", 7: "image8", 8: "image9"
}

# Load images with error handling
images = {}
image_paths = {
    "image1": "water.jpeg",
    "image2": "washroom.jpeg", 
    "image3": "food.jpeg",
    "image4": "emergency.jpeg",
    "image5": "medicine.jpeg",
    "image6": "cloth.webp",
    "image7": "stature.webp",
    "image8": "lights.jpeg",
    "image9": "un.png"
}

# Load and resize images
for img_key, img_path in image_paths.items():
    try:
        img = cv2.imread(img_path)
        if img is not None:
            images[img_key] = cv2.resize(img, (300, 300))
        else:
            # Create placeholder image
            placeholder = np.zeros((300, 300, 3), np.uint8)
            cv2.putText(placeholder, img_key, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            images[img_key] = placeholder
            print(f"Warning: Image {img_path} not found. Using placeholder.")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        # Create placeholder image
        placeholder = np.zeros((300, 300, 3), np.uint8)
        cv2.putText(placeholder, img_key, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        images[img_key] = placeholder

# Image names for speech
image_names = {
    "image1": "Water",
    "image2": "Washroom",
    "image3": "Food", 
    "image4": "Emergency",
    "image5": "Medicine",
    "image6": "Change my clothes",
    "image7": "Change my position",
    "image8": "Lights",
    "image9": "Unpleasant environment"
}

# TTS function
def speak(text):
    """Speak text using TTS in a separate thread"""
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        print(f"TTS Error: Could not speak '{text}'")

# Drawing functions
font = cv2.FONT_HERSHEY_PLAIN

def draw_images_grid(letter_index):
    """Draw the 3x3 grid of images"""
    image_grid[:] = (26, 26, 26)  # Reset background
    
    num_columns = 3
    for i in range(len(keys_set_1)):
        x = (i % num_columns) * 300
        y = (i // num_columns) * 300
        
        img_key = keys_set_1[i]
        if img_key in images:
            img = images[img_key]
            
            # Highlight selected image
            if i == letter_index:
                cv2.rectangle(image_grid, (x, y), (x + 300, y + 300), (255, 255, 255), 5)
            
            # Place image
            image_grid[y:y + 300, x:x + 300] = img
            
            # Add border
            cv2.rectangle(image_grid, (x, y), (x + 300, y + 300), (100, 100, 100), 2)

def midpoint(p1, p2):
    """Calculate midpoint between two points"""
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    """Calculate eye aspect ratio to detect blinking"""
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    
    if ver_line_length == 0:
        return 0
    
    ratio = hor_line_length / ver_line_length
    return ratio

def eyes_contour_points(facial_landmarks):
    """Get eye contour points"""
    left_eye = []
    right_eye = []
    
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
        
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
        
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks, gray_frame):
    """Calculate gaze ratio to detect eye movement direction"""
    left_eye_region = np.array([
        (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
        (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
        (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
        (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
        (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
        (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
    ], np.int32)
    
    # Create mask for the eye region
    mask = np.zeros_like(gray_frame)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
    
    # Get eye region boundaries
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    
    # Extract and threshold the eye region
    gray_eye = eye[min_y:max_y, min_x:max_x]
    if gray_eye.size == 0:
        return 1
        
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    
    # Split the eye into left and right sides
    height, width = threshold_eye.shape
    if width < 2:
        return 1
        
    left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    
    right_side_threshold = threshold_eye[0:height, int(width/2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
        
    return gaze_ratio

def update_board(text):
    """Update the text board display"""
    board[:] = 255  # Clear board
    
    # Format text with word wrapping
    max_chars_per_line = 35
    lines = []
    words = text.split(' ')
    
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars_per_line:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())
    
    # Draw text lines on board
    y0, dy = 60, 60
    for i, line in enumerate(lines):
        if i * dy + y0 < board.shape[0]:  # Check if line fits on board
            cv2.putText(board, line, (20, y0 + i * dy), font, 4, 0, 3, cv2.LINE_AA)

# Initialize variables
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 3
frames_active_letter = 30
text = ""
no_face_frames = 0
max_no_face_frames = 150
cooldown_frames = 25
cooldown_counter = 0

# Pattern detection variables
blink_pattern = []
pattern_start_time = 0
pattern_timeout = 3
last_blink_time = time.time()
blink_warning_shown = False

# Emergency patterns
recognized_patterns = {
    '111': 'emergency',
    '11': 'water', 
    '101': 'washroom',
    '1001': 'food'
}

def handle_pattern_action(action):
    """Handle recognized blink patterns"""
    global text
    if action == 'emergency':
        text += " EMERGENCY!"
        threading.Thread(target=speak, args=("Emergency!",)).start()
        if siren_sound:
            siren_sound.play()
    elif action == 'water':
        text += " Water"
        threading.Thread(target=speak, args=("Water",)).start()
        if sound:
            sound.play()
    elif action == 'washroom':
        text += " Washroom"
        threading.Thread(target=speak, args=("Washroom",)).start()
        if sound:
            sound.play()
    elif action == 'food':
        text += " Food"
        threading.Thread(target=speak, args=("Food",)).start()
        if sound:
            sound.play()

# Main loop
print("Vision Board Application Started")
print("Blink patterns: 111=Emergency, 11=Water, 101=Washroom, 1001=Food")
print("Look at images and blink to select. Press ESC to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        frame = cv2.flip(frame, 1)  # Mirror the frame
        rows, cols, _ = frame.shape
        frames += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw loading bar background
        frame[rows - 50:rows, 0:cols] = (255, 255, 255)
        
        # Get current active image
        active_image = keys_set_1.get(letter_index, "")
        
        # Face detection
        faces = detector(gray)
        
        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames >= max_no_face_frames:
                print("No face detected for too long. Exiting...")
                break
                
            # Display countdown
            countdown = max(1, (max_no_face_frames - no_face_frames) // 30)
            cv2.putText(frame, "No Person Detected", (45, 200), font, 4, (255, 0, 0), 3)
            cv2.putText(frame, f"Closing in {countdown} seconds", (30, 150), font, 4, (0, 0, 255), 3)
            
        elif len(faces) > 1:
            cv2.putText(frame, "Multiple faces detected", (30, 150), font, 4, (255, 0, 0), 3)
            cv2.putText(frame, "Please ensure only one person", (30, 200), font, 3, (255, 0, 0), 3)
            
        else:
            no_face_frames = 0
            face = faces[0]
            landmarks = predictor(gray, face)
            
            # Get eye contours
            left_eye, right_eye = eyes_contour_points(landmarks)
            
            # Calculate blinking ratio
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            
            # Draw eye contours
            cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
            
            # Pattern detection for emergency commands
            current_time = time.time()
            if blinking_ratio > 5:  # Eyes closed
                if len(blink_pattern) == 0 or current_time - pattern_start_time > 0.5:
                    blink_pattern.append('1')
                    pattern_start_time = current_time
                    last_blink_time = current_time
            else:  # Eyes open
                if len(blink_pattern) > 0 and current_time - pattern_start_time > 0.5:
                    blink_pattern.append('0')
            
            # Check for completed patterns
            pattern_str = ''.join(blink_pattern)
            for pattern, action in recognized_patterns.items():
                if pattern_str.endswith(pattern):
                    handle_pattern_action(action)
                    blink_pattern = []
                    break
            
            # Clear old patterns
            if current_time - pattern_start_time > pattern_timeout:
                blink_pattern = []
            
            # Blink warning system
            if current_time - last_blink_time > 10 and not blink_warning_shown:
                print("Warning: No blinking detected for 10 seconds")
                threading.Thread(target=speak, args=("Are you okay?",)).start()
                if siren_sound:
                    siren_sound.play()
                blink_warning_shown = True
            elif current_time - last_blink_time <= 10:
                blink_warning_shown = False
            
            # Handle cooldown period
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                # Detect blinking to select image
                if blinking_ratio > 5:
                    blinking_frames += 1
                    frames -= 1
                    
                    # Show green eyes when closed
                    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
                    
                    # Select image
                    if blinking_frames == frames_to_blink:
                        if active_image in image_names:
                            if active_image == "image4":  # Emergency
                                text += " " + image_names[active_image]
                                threading.Thread(target=speak, args=(image_names[active_image],)).start()
                                if siren_sound:
                                    siren_sound.play()
                            else:
                                text += " " + image_names[active_image]
                                threading.Thread(target=speak, args=(image_names[active_image],)).start()
                                if sound:
                                    sound.play()
                        
                        # Reset counters
                        frames = 0
                        blinking_frames = 0
                        cooldown_counter = cooldown_frames
                else:
                    blinking_frames = 0
        
        # Update image selection
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0
        
        if letter_index >= len(keys_set_1):
            letter_index = 0
        
        # Draw the image grid
        draw_images_grid(letter_index)
        
        # Update text board
        update_board(text)
        
        # Draw blinking progress bar
        if blinking_frames > 0:
            percentage_blinking = min(blinking_frames / frames_to_blink, 1.0)
            loading_x = int(cols * percentage_blinking)
            cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
        
        # Display current selection info
        if active_image in image_names:
            cv2.putText(frame, f"Current: {image_names[active_image]}", 
                       (10, 30), font, 2, (0, 255, 0), 2)
        
        # Show frames
        cv2.imshow("Vision Board - Camera", frame)
        cv2.imshow("Vision Board - Images", image_grid)
        cv2.imshow("Vision Board - Text", board)
        
        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # Reset text
            text = ""
            print("Text reset")

except KeyboardInterrupt:
    print("\nApplication interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Vision Board Application Closed")