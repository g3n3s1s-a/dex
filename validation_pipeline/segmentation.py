import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from scipy.ndimage import binary_fill_holes
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Folders (defined at module level) ---
# folder where the images post resizing are
folder = "validation/"
output_folder = "segmented_images/"


def segment_hand_skin(image):
    """skin color segmentation"""
    image_resized = resize(image, (275, 275), anti_aliasing=True)
    img_uint8 = (image_resized * 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
    
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    mask = mask.astype(bool)
    
    mask = closing(mask, disk(5))
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, max_size=499)
    mask = remove_small_holes(mask, max_size=500)
    
    return mask.astype(np.uint8), image_resized


def process_single_image(filename):
    """Process one image - for parallel execution"""
    try:
        path = os.path.join(folder, filename)
        image = io.imread(path)
        
        mask, image_resized = segment_hand_skin(image)
        segmented = image_resized * mask[..., np.newaxis]
        
        save_path = os.path.join(output_folder, filename)
        io.imsave(save_path, (segmented * 255).astype(np.uint8))
        
        return True, filename
    except Exception as e:
        return False, f"{filename}: {str(e)}"


if __name__ == '__main__':
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files
    all_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".jpg")]
    total = len(all_files)
    
    print(f"Processing {total:,} images using parallel processing...")
    print(f"Using {os.cpu_count()} CPU cores\n")
    
    processed = 0
    errors = []
    
    # Use all available CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, f): f for f in all_files}
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=total, desc="Segmenting", unit="img"):
            success, result = future.result()
            if success:
                processed += 1
            else:
                errors.append(result)
    
    print(f"   Processed: {processed:,} images")
    print(f"   Errors: {len(errors):,} images")
    
    if errors:
        with open("errors.log", "w") as f:
            f.write("\n".join(errors))
        print(f"   Error log saved to errors.log")
