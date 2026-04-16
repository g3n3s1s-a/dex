import os
import numpy as np
import cv2
import mediapipe as mp
from skimage import io
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Folders ---
folder = "../dataset/prj03_resized_train/"
output_folder = "mediapipe_hand_output/"

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def extract_hand_crop(image):
    """
    Detects hand using MediaPipe and returns a cropped 128x128 image.
    Falls back to resized image if no hand is detected.
    """

    # Ensure uint8 format
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    h, w, _ = image_uint8.shape

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # Fallback: no hand detected
    if not results.multi_hand_landmarks:
        return cv2.resize(image_uint8, (128, 128))

    landmarks = results.multi_hand_landmarks[0]

    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]

    x_min, x_max = max(min(xs) - 30, 0), min(max(xs) + 30, w)
    y_min, y_max = max(min(ys) - 30, 0), min(max(ys) + 30, h)

    cropped = image_uint8[y_min:y_max, x_min:x_max]

    if cropped.size == 0:
        return cv2.resize(image_uint8, (128, 128))

    cropped = cv2.resize(cropped, (128, 128))

    return cropped


def process_single_image(filename):
    """Process one image for multiprocessing."""

    try:
        path = os.path.join(folder, filename)
        image = io.imread(path)

        # Optional resize first (keeps consistency with your old pipeline)
        image_resized = resize(image, (275, 275), anti_aliasing=True)
        image_uint8 = (image_resized * 255).astype(np.uint8)

        # MediaPipe hand extraction
        processed = extract_hand_crop(image_uint8)

        # Save output
        save_path = os.path.join(output_folder, filename)
        io.imsave(save_path, processed)

        return True, filename

    except Exception as e:
        return False, f"{filename}: {str(e)}"


if __name__ == '__main__':

    os.makedirs(output_folder, exist_ok=True)

    all_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".jpg")]
    total = len(all_files)

    print(f"Processing {total:,} images using MediaPipe...")
    print(f"Using {os.cpu_count()} CPU cores\n")

    processed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, f): f for f in all_files}

        for future in tqdm(as_completed(futures), total=total, desc="Processing", unit="img"):
            success, result = future.result()

            if success:
                processed += 1
            else:
                errors.append(result)

    print(f"\nProcessed: {processed:,} images")
    print(f"Errors: {len(errors):,} images")

    if errors:
        with open("errors.log", "w") as f:
            f.write("\n".join(errors))

        print("Error log saved to errors.log")
