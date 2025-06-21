import cv2
import numpy as np

# Set up image grid dimensions (3x3 grid)
num_rows = 3
num_cols = 3  # Total columns for all images

# Load unique images for the grid
images_left = [
    cv2.imread("water.jpeg"),
    cv2.imread("washroom.jpeg"),
    cv2.imread("food.jpeg"),
    cv2.imread("emergency.jpeg"),
    cv2.imread("medicine.jpeg")
]

images_right = [
    cv2.imread("cloth.webp"),
    cv2.imread("stature.webp"),
    cv2.imread("lights.jpeg"),
    cv2.imread("un.png")
]

# Combine images from both left and right
all_images = images_left + images_right

# Adjust the number of images to fit the 3x3 grid (if there are fewer than 9 images, we can duplicate or add placeholders)
all_images = all_images[:9]  # Limit the images to 9 for a 3x3 grid

# Resize the images to fit within the grid, preserving aspect ratio
def resize_image_to_fit(image, target_width, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Create a new blank canvas with the target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Place the resized image on the canvas (centered)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    
    return canvas

# Reduce the image and frame sizes
img_width = 300  # Reduced width of each image
img_height = 300  # Reduced height of each image

# Resize the images to fit within the grid, preserving aspect ratio
all_images_resized = [resize_image_to_fit(img, img_width, img_height) for img in all_images]

# Calculate the total frame size based on grid dimensions and resized images
frame_width = img_width * num_cols
frame_height = img_height * num_rows

# Create black background for the combined frame
combined_frame = np.zeros((frame_height, frame_width, 3), np.uint8)  # Total combined frame

# Function to place images on the grid
def draw_grid():
    for i, img in enumerate(all_images_resized):
        row = i // num_cols
        col = i % num_cols
        combined_frame[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img

# Draw the images in the grid
draw_grid()

# Show the combined grid
cv2.imshow("Combined Grid", combined_frame)

# Wait until a key is pressed to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
