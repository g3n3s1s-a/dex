import matplotlib.pyplot as plt
from skimage import io, color, feature
from skimage.segmentation import chan_vese, watershed
from skimage.filters import sobel
from skimage.transform import resize
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from scipy.ndimage import binary_fill_holes
import numpy as np
import cv2


def extract_mask_edge_features(image, target_size=(275, 275)):
    """
    Returns a skin mask and an edge map from the image.
    """
    # Resize and convert to uint8
    image_resized = resize(image, target_size, anti_aliasing=True)
    img_uint8 = (image_resized * 255).astype(np.uint8)

    # Convert to HSV and YCrCb
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)

    # Skin detection thresholds
    mask_hsv = cv2.inRange(hsv, np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb).astype(bool)

    # Morphological operations
    mask = closing(mask, disk(5))
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, max_size=499)
    mask = remove_small_holes(mask, max_size=500)

    # Edge map using Canny
    gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_uint8, 100, 200) > 0

    return mask.astype(np.uint8), edges.astype(np.uint8)


def compare_segmentation_methods(image_path):
    # Load and grayscale
    image = io.imread(image_path)
    gray = color.rgb2gray(image)

    # --- 1. Chan-Vese ---
    cv_mask = chan_vese(gray, mu=0.3, lambda1=1, lambda2=1,
                        tol=1e-3, max_num_iter=300)
    cv_mask = cv_mask.astype(bool)

    # --- 2. Watershed ---
    gradient = sobel(gray)
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[gray < gray.mean()] = 1
    markers[gray > gray.mean()] = 2
    ws_mask = watershed(gradient, markers) == 2

    # --- 3. Skin + edges ---
    skin_mask, edges = extract_mask_edge_features(image)
    combined = np.maximum(skin_mask, edges)

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(cv_mask, cmap='gray')
    axes[1].set_title("Chan-Vese")
    axes[1].axis('off')

    axes[2].imshow(ws_mask, cmap='gray')
    axes[2].set_title("Watershed")
    axes[2].axis('off')

    axes[3].imshow(combined, cmap='gray')
    axes[3].set_title("Color-based skin segmentation")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_comparison.png", dpi=150)
    plt.show()


# Example usage
compare_segmentation_methods("../dataset/prj03_resized_train/P6_N_522.jpg")
