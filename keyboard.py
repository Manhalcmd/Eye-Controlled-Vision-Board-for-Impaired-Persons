import cv2
import numpy as np

# Initialize the camera capture
# cap = cv2.VideoCapture(0)

# Create the empty board for text display
board = np.zeros((450, 1400), np.uint8)
board[:] = 255

# Create the virtual keyboard display
keyboard = np.zeros((900, 1200, 3), np.uint8)

# Set of keys for the keyboard
keys_set_2 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
    5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
    10: "A", 11: "S", 12: "D", 13: "F", 14: "G",
    15: "H", 16: "J", 17: "K", 18: "L",
    19: "Z", 20: "X", 21: "C", 22: "V", 23: "B",
    24: "N", 25: "M", 26: "_", 27: "Exit"
}

font = cv2.FONT_HERSHEY_PLAIN

# Draw the keyboard UI
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

# Main loop for displaying the keyboard
letter_index = 0
while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break

    # rows, cols, _ = frame.shape
    keyboard[:] = (26, 26, 26)  # Reset keyboard

    # Draw keyboard UI
    for i in range(len(keys_set_2)):
        draw_letters2(i, keys_set_2[i], i == letter_index)

    # Display the camera frame and virtual keyboard
    # cv2.imshow("Frame", frame)
    cv2.imshow("Virtual Keyboard", keyboard)

    key = cv2.waitKey(1)
    if key == 27:  # Escape key to exit
        break
    if key == 2490368:  # Arrow Up key to move up in the keyboard
        letter_index = (letter_index - 9) % len(keys_set_2)
    elif key == 2621440:  # Arrow Down key to move down in the keyboard
        letter_index = (letter_index + 9) % len(keys_set_2)
    elif key == 2424832:  # Arrow Left key to move left in the keyboard
        letter_index = (letter_index - 1) % len(keys_set_2)
    elif key == 2555904:  # Arrow Right key to move right in the keyboard
        letter_index = (letter_index + 1) % len(keys_set_2)

# cap.release()
cv2.destroyAllWindows()
