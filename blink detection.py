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

# Load sounds
sound = pyglet.media.load("sound.wav", streaming=False)
left_sound = pyglet.media.load("left.wav", streaming=False)
right_sound = pyglet.media.load("right.wav", streaming=False)
siren_sound = pyglet.media.load("siren.wav", streaming=False)


cap = cv2.VideoCapture(0)
board = np.zeros((450, 1400), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

keyboard = np.zeros((1170, 1170, 3), np.uint8) 
keys_set_1 = {0: "image1", 1: "image2", 2: "image3", 3: "image4", 4: "image5", 5: "image6", 6: "image7", 7: "image8", 8: "image9"}

keys_set_2 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
    5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
    10: "A", 11: "S", 12: "D", 13: "F", 14: "G",
    15: "H", 16: "J", 17: "K", 18: "L",
    19: "Z", 20: "X", 21: "C", 22: "V", 23: "B",
    24: "N", 25: "M", 26: "_", 27: "Exit"
}

# Load images for keys_set_1
images = {
    "image1": cv2.imread("water.jpeg"),
    "image2": cv2.imread("washroom.jpeg"),
    "image3": cv2.imread("food.jpeg"),
    "image4": cv2.imread("emergency.jpeg"),
    "image5": cv2.imread("medicine.jpeg"),
    "image6": cv2.imread("cloth.webp"),
    "image7": cv2.imread("stature.webp"),
    "image8": cv2.imread("lights.jpeg"),
    "image9": cv2.imread("un.png")
}

# Image names for keys_set_1
image_names = {
    "image1": "Water",
    "image2": "Washroom",
    "image3": "Food",
    "image4": "Emercency",
    "image5": "medicine",
    "image6": "change my cloths",
    "image7": "change my stature",
    "image8": "lights",
    "image9": "unpleasant environment"
}
def speak(text):
    engine.say(text)
    engine.runAndWait()


font = cv2.FONT_HERSHEY_PLAIN

def draw_letters1(letter_index, text, letter_light):
    num_columns = 3  # Number of columns for images
    x = (letter_index % num_columns) * 300  # Adjusted for 3 images per row, increased x-spacing
    y = (letter_index // num_columns) * 250  # Adjusted for 300 pixels height per row

    width = 300 # Adjusted width
    height = 250  # Adjusted height
    th = 5  # thickness (adjust as needed)

    if text.startswith("image"):
        if text in images:
            img = images[text]
            img = cv2.resize(img, (width - th * 2, height - th * 2))

            if letter_light:
                cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
            else:
                cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)

            img_x = x + th
            img_y = y + th
            keyboard[img_y:img_y + img.shape[0], img_x:img_x + img.shape[1]] = img
        else:
            print(f"Image {text} not found in images dictionary")
    else:
        print("Invalid text for draw_letters1")

keyboard = np.zeros((900, 1200, 3), np.uint8) 

def draw_letters2(letter_index, text, letter_light):
    rows = 9  # Number of rows
    cols = 9  # Number of columns
    x = (letter_index % cols) * 130  # Adjusted for 9 letters per row
    y = (letter_index // cols) * 130  # Adjusted for 160 pixels height per row

    width = 130  # Adjusted width
    height = 130  # Adjusted height
    th = 4  # thickness

    if text in keys_set_2.values():
        if letter_light:
            cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
            cv2.putText(keyboard, text, (x + 20, y + 100), font, 3, (51, 51, 51), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
            cv2.putText(keyboard, text, (x + 20, y + 100), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        print("Invalid text for draw_letters2")

def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4  # thickness lines
    cv2.line(keyboard, (int(cols / 2) - int(th_lines / 2), 0), (int(cols / 2) - int(th_lines / 2), rows), (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(cols / 2), 300), font, 6, (255, 255, 255), 5)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def eyes_contour_points(facial_landmarks):
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

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    mask = np.zeros_like(gray)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 1
frames_active_letter = 30

# Text and keyboard settings
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0

no_face_frames = 0
max_no_face_frames = 150

cooldown_frames = 25  # Number of frames to wait after selecting a key
cooldown_counter = 0  # Counter to track the cooldown period

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rows, cols, _ = frame.shape
    keyboard[:] = (26, 26, 26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)
    
    if select_keyboard_menu:
        draw_menu()

    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set_1
    else:
        keys_set = keys_set_2
    active_letter = keys_set.get(letter_index, "")

    # Face detection
    faces = detector(gray)
    if len(faces) == 0:
        no_face_frames += 1
        if no_face_frames == max_no_face_frames:
            break
        # Display countdown on screen
        countdown = (max_no_face_frames - no_face_frames) // 30 + 1  # Convert frames to seconds
        cv2.putText(frame, "No Person Detected", (45, 200), font, 4, (255, 0, 0), thickness=3)
        cv2.putText(frame, f"Closing in {countdown} seconds", (30, 150), font, 4, (0, 0, 255), thickness=3)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        continue
    else:
        no_face_frames = 0  # Reset the no face frames counter

    # Handle case when more than one face is detected
    if len(faces) > 1:
        cv2.putText(frame, "Multiple faces", (30, 150), font, 4, (255, 0, 0), thickness=3)
        cv2.putText(frame, "detected", (45, 200), font, 4, (255, 0, 0), thickness=3)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        continue

    # Proceed with single face processing
    face = faces[0]
    landmarks = predictor(gray, face)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye, right_eye = eyes_contour_points(landmarks)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Eyes color
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

        if select_keyboard_menu:
            # Detecting gaze to select Left or Right keyboard
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            if gaze_ratio <= 0.9:
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    right_sound.play()
                    frames = 0
                    keyboard_selection_frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
            else:
                keyboard_selected = "left"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    left_sound.play()
                    frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0

        else:
            # Handle the cooldown period
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                # Detect the blinking to select the key that is lighting up
                if blinking_ratio > 5:
                    blinking_frames += 1
                    frames -= 1

                    # Show green eyes when closed
                    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                    # Typing letter
                    if blinking_frames == frames_to_blink:
                        if keyboard_selected == "left" and active_letter.startswith("image"):
                            if active_letter == "image4":
                                text += " " + image_names[active_letter]  # Append the image name to the text
                                threading.Thread(target=speak, args=(image_names[active_letter],)).start()
                                siren_sound.play()
                            else:
                                text += " " + image_names[active_letter]
                                threading.Thread(target=speak, args=(image_names[active_letter],)).start()
                                sound.play()
                        elif keyboard_selected == "right":
                            if active_letter == "Exit":
                                select_keyboard_menu = True  # Only close the keyboard menu if CLOSE is selected
                            else:
                                if active_letter != "<" and active_letter != "_":
                                    text += active_letter
                                    sound.play()
                                if active_letter == "_":
                                    text += " "
                        
                        # Remove the line resetting letter_index to 0
                        # letter_index = 0  # <-- REMOVE or COMMENT OUT THIS LINE

                        # Reset frames to avoid rapid reselection
                        frames = 0

                        # Reset blinking_frames after selection
                        blinking_frames = 0
                        
                        # Start cooldown period
                        cooldown_counter = cooldown_frames

                        if keyboard_selected != "right":  # Reset menu only if not using the right keyboard
                            select_keyboard_menu = True
                else:
                    blinking_frames = 0

    # Display letters on the keyboard
    if not select_keyboard_menu:
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0

        if keyboard_selected == "left":
            if letter_index >= len(keys_set_1):  # Adjust condition to reset letter_index for keys_set_1
                letter_index = 0
            for i in range(len(keys_set_1)):
                draw_letters1(i, keys_set_1[i], i == letter_index)
        elif keyboard_selected == "right":
            if letter_index >= len(keys_set_2):  # Adjust condition to reset letter_index for keys_set_2
                letter_index = 0
            for i in range(len(keys_set_2)):
                draw_letters2(i, keys_set_2[i], i == letter_index)

    # Show the text we're writing on the board
    max_chars_per_line = 35  # Adjust based on your board width and font size
    lines = []
    words = text.split(' ')  # Split text into words by spaces

    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars_per_line:  # +1 for space
            current_line += word + " "
        else:
            lines.append(current_line.strip())  # Append the current line and strip trailing space
            current_line = word + " "
    lines.append(current_line.strip())  # Append the last line and strip trailing space

    y0, dy = 60, 60  # Adjust starting position and line height
    for i, line in enumerate(lines):
        cv2.putText(board, line, (20, y0 + i * dy), font, 4, 0, 3, cv2.LINE_AA)

    # Blinking loading bar
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
