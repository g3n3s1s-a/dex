import os
import numpy as np
import cv2
from skimage import io
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# --- GLOBALS (needed for multiprocessing) ---
sam_checkpoint = "sam_vit_b.pth"   # <-- download this first
model_type = "vit_b"

predictor = None


def init_sam():
    """Initialize SAM model (called once per worker)"""
    global predictor
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)


def segment(image):
    """Segment using SAM"""
    global predictor

    image_resized = resize(image, (275, 275), anti_aliasing=True)
    image_uint8 = (image_resized * 255).astype(np.uint8)

    predictor.set_image(image_uint8)

    h, w, _ = image_uint8.shape

    # Use center point as prompt (works well for hands)
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # pick best mask
    best_mask = masks[np.argmax(scores)]

    return best_mask.astype(np.uint8), image_resized


def process_single_image(filename):
    try:
        path = os.path.join(folder, filename)
        image = io.imread(path)

        mask, image_resized = segment(image)

        segmented = image_resized * mask[..., np.newaxis]

        save_path = os.path.join(output_folder, filename)
        io.imsave(save_path, (segmented * 255).astype(np.uint8))

        return True, filename

    except Exception as e:
        return False, f"{filename}: {str(e)}"

def main(folder,output_folder)

    #folder = "../dataset/prj03_resized_train/"
    #output_folder = "sam_segmented_output/"

    os.makedirs(output_folder, exist_ok=True)

    all_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".jpg")]
    total = len(all_files)

    print(f"Processing {total:,} images with SAM...")
    print(f"Using {os.cpu_count()} CPU cores\n")

    processed = 0
    errors = []

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_sam
    ) as executor:

        futures = {executor.submit(process_single_image, f): f for f in all_files}

        for future in tqdm(as_completed(futures), total=total, desc="Segmenting", unit="img"):
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
if __name__ == '__main__':
    folder = "../dataset/prj03_resized_train/"
    output_folder = "sam_segmented_output/"
    main(folder,output_folder)
