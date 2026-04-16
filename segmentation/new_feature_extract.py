import os
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# --- Folder with MediaPipe outputs ---
image_folder = "mediapipe_hand_output/"
output_file = "features_hog.pkl"

# --- Parameters ---
IMG_SIZE = (128, 128)


def extract_hog_features(image):
    """
    Extract HOG features from MediaPipe-cropped hand image
    """

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    # Resize for consistency
    gray = resize(gray, IMG_SIZE, anti_aliasing=True)

    # HOG feature extraction
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

    return features


def process_single_image(filename):
    """
    Process one image (for multiprocessing)
    Returns: (success, feature_vector, label, filename_or_error)
    """

    try:
        path = os.path.join(image_folder, filename)

        img = Image.open(path)
        image = np.array(img)
        img.close()

        # Extract features
        feature_vector = extract_hog_features(image)

        # Extract label from filename (e.g., P6_N_522.jpg → N)
        label = filename.split("_")[1]

        return True, feature_vector, label, filename

    except Exception as e:
        return False, None, None, f"{filename}: {str(e)}"


if __name__ == "__main__":

    # --- Load files ---
    all_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(".jpg")
    ])

    total_files = len(all_files)

    print(f"Extracting HOG features from {total_files:,} images...")
    print(f"Image size: {IMG_SIZE}")
    print(f"Using {os.cpu_count()} CPU cores\n")

    features = []
    labels = []
    errors = []
    successful_files = []

    # --- Parallel processing ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(process_single_image, f): f
            for f in all_files
        }

        for future in tqdm(as_completed(futures),
                           total=total_files,
                           desc="Extracting",
                           unit="img"):

            success, feat, label, info = future.result()

            if success:
                features.append(feat)
                labels.append(label)
                successful_files.append(info)
            else:
                errors.append(info)

    # --- Convert to numpy arrays ---
    features = np.array(features)
    labels = np.array(labels)

    print("\nNormalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # --- Summary ---
    print("\n✅ Feature extraction complete!")
    print(f"   Processed: {len(features):,}/{total_files:,}")
    print(f"   Errors: {len(errors):,}")
    print(f"   Feature shape: {features.shape}")
    print(f"   Classes: {sorted(np.unique(labels))}")

    # --- Save features ---
    with open(output_file, "wb") as f:
        pickle.dump({
            "features": features,
            "labels": labels,
            "scaler": scaler
        }, f)

    print(f"\nSaved to {output_file}")

    # --- Error log ---
    if errors:
        with open("feature_errors.log", "w") as f:
            f.write("\n".join(errors))
        print("Error log saved to feature_errors.log")
