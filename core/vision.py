import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_leaf(image):
    """
    Validates if the input image is a valid crop leaf image.
    Uses HSV color filtering and Canny edge detection.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image in RGB format

    Returns:
    --------
    tuple: (bool, str)
        - Boolean indicating if image is valid
        - Error message if invalid, empty string if valid
    """
    # Convert RGB to HSV for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define HSV ranges for green, yellow, and brown hues (leaf colors)
    # Green range (healthy leaves)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Yellow range (diseased/aging leaves)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    # Brown range (diseased/dead leaves)
    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([20, 255, 200])

    # Create masks for each color range
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Combine all masks
    combined_mask = cv2.bitwise_or(mask_green, mask_yellow)
    combined_mask = cv2.bitwise_or(combined_mask, mask_brown)

    # Calculate percentage of leaf-colored pixels
    total_pixels = image.shape[0] * image.shape[1]
    leaf_pixels = np.sum(combined_mask > 0)
    leaf_percentage = (leaf_pixels / total_pixels) * 100

    # Check if leaf color percentage is sufficient (15-20% minimum)
    if leaf_percentage < 15:
        return False, "Invalid Image: Please upload a clear photo of a crop leaf."

    # Perform Canny edge detection to ensure image has texture
    # Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    # Calculate edge density
    edge_pixels = np.sum(edges > 0)
    edge_percentage = (edge_pixels / total_pixels) * 100

    # Check if image has sufficient edges (not blank sky or solid color)
    if edge_percentage < 2:  # At least 2% of pixels should be edges
        return False, "Invalid Image: Please upload a clear photo of a crop leaf."

    return True, ""
