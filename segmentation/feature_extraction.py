import os
import numpy as np
from skimage import io, color, feature
from skimage.transform import resize
from skimage.filters import gaussian
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# --- Folders ---
segmented_folder = "skin_segmented_output/"
features_file = "features_mask_edges.pkl"

# --- Parameters ---
ORIGINAL_SIZE = (275, 275)   # original image size
DOWNSAMPLE_SIZE = (64, 64)   # reduced size for features
EDGE_WEIGHT = 2.0            # emphasize edges more than mask


def extract_mask_edge_features(segmented_image):
    """
    Extract mask + edge features with dimensionality reduction
    """
    # --- Convert to grayscale ---
    if len(segmented_image.shape) == 3:
        gray = color.rgb2gray(segmented_image)
    else:
        gray = segmented_image

    # --- Ensure consistent size (safety) ---
    resized = resize(gray, ORIGINAL_SIZE, anti_aliasing=True)

    # --- 1. Create binary mask ---
    mask = (resized > 0.05).astype(np.float32)

    # --- 2. Edge detection ---
    smoothed = gaussian(resized, sigma=1.0)
    edges = feature.canny(
        smoothed,
        sigma=1.0,
        low_threshold=0.1,
        high_threshold=0.2
    ).astype(np.float32)

    # --- 3. Downsample (CRITICAL for SVM performance) ---
    mask_small = resize(mask, DOWNSAMPLE_SIZE, anti_aliasing=True)
    edges_small = resize(edges, DOWNSAMPLE_SIZE, anti_aliasing=True)

    # --- 4. Flatten + combine ---
    mask_flat = mask_small.flatten()
    edges_flat = edges_small.flatten()

    combined_features = np.concatenate([
        mask_flat,
        edges_flat * EDGE_WEIGHT   # emphasize edges
    ])

    return combined_features


def process_single_image(filename):
    """
    Process one image - for parallel execution
    Returns: (success, feature_vector, label, filename_or_error)
    """
    try:
        path = os.path.join(segmented_folder, filename)
        
        # Read with PIL (faster and more robust)
        pil_img = Image.open(path)
        image = np.array(pil_img)
        pil_img.close()
        
        # Extract features
        feature_vector = extract_mask_edge_features(image)
        
        # Extract label (e.g., "P6_N_522.jpg" → "N")
        label = filename.split("_")[1]
        
        return True, feature_vector, label, filename
        
    except Exception as e:
        return False, None, None, f"{filename}: {str(e)}"


if __name__ == '__main__':
    # --- Load files ---
    all_files = sorted([f for f in os.listdir(segmented_folder) if f.endswith(".jpg")])
    total_files = len(all_files)
    
    print(f"Extracting features from {total_files:,} images...")
    print(f"Original size: {ORIGINAL_SIZE}")
    print(f"Downsampled size: {DOWNSAMPLE_SIZE}")
    print(f"Feature dimension: {DOWNSAMPLE_SIZE[0]*DOWNSAMPLE_SIZE[1]*2:,}")
    print(f"Using {os.cpu_count()} CPU cores\n")
    
    features = []
    labels = []
    errors = []
    successful_files = []
    
    # --- Parallel processing ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, f): f for f in all_files}
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=total_files, desc="Extracting", unit="img"):
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
    print(f"\n✅ Feature extraction complete!")
    print(f"   Successfully processed: {len(features):,}/{total_files:,}")
    print(f"   Errors: {len(errors):,}")
    print(f"   Features shape: {features.shape}")
    print(f"   Feature dimension per image: {features.shape[1]:,}")
    print(f"   Classes: {sorted(np.unique(labels))}")
    
    # --- Save ---
    with open(features_file, 'wb') as f:
        pickle.dump({
            'features': features,
            'labels': labels,
            'scaler': scaler
        }, f)
    
    print(f"\nSaved to {features_file}")
    
    # --- Save error log if any ---
    if errors:
        with open("feature_extraction_errors.log", "w") as f:
            f.write("\n".join(errors))
        print(f"Error log saved to feature_extraction_errors.log")
    
    # --- Visualize a few samples ---
    print("\n Visualizing sample images...")
    VISUALIZE_SAMPLES = 3
    
    for i in range(min(VISUALIZE_SAMPLES, len(successful_files))):
        filename = successful_files[i]
        path = os.path.join(segmented_folder, filename)
        image = io.imread(path)
        
        # Re-extract for visualization
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        
        resized = resize(gray, ORIGINAL_SIZE, anti_aliasing=True)
        mask = (resized > 0.05).astype(np.float32)
        smoothed = gaussian(resized, sigma=1.0)
        edges = feature.canny(smoothed, sigma=1.0, low_threshold=0.1, high_threshold=0.2).astype(np.float32)
        mask_small = resize(mask, DOWNSAMPLE_SIZE, anti_aliasing=True)
        edges_small = resize(edges, DOWNSAMPLE_SIZE, anti_aliasing=True)
        
        label = filename.split("_")[1]
        
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        
        axes[0].imshow(image)
        axes[0].set_title("Segmented")
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask (Full)")
        axes[1].axis('off')
        
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title("Edges (Full)")
        axes[2].axis('off')
        
        axes[3].imshow(mask_small, cmap='gray')
        axes[3].set_title("Mask (Downsampled)")
        axes[3].axis('off')
        
        axes[4].imshow(edges_small, cmap='gray')
        axes[4].set_title("Edges (Downsampled)")
        axes[4].axis('off')
        
        plt.suptitle(f"{filename} | Label: {label}")
        plt.tight_layout()
        plt.show()
