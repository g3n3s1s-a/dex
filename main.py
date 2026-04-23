# preprocessing is in this file
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from scipy.ndimage import binary_fill_holes
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from skimage import color, feature
from skimage.transform import resize
from skimage.filters import gaussian
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib

import prj03
from segmentation import segmentation,feature_extraction, svm_train
def run_pipeline():
    print("--- Starting Pipeline ---")
    print("--- Starting Preprocessing Step ---")
    input_dir = 'dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/asl_processed/train'
    preproc_dir = 'training_processed'
    prj03.preprocessing(input_dir,preproc_dir)


    # Step 2: segmentation
    print("--- Starting Segmentation Step ---")
    segmented_dir = 'training_segmented'
    segmentation.main(preproc_dir,segmented_dir)

    # step 3 feature extraction
    feature_file_name = "features_mask_edges.pkl"
    print("-- Starting Feature Extraction Step ---")
    feature_extraction.main(segmented_dir,feature_file_name)

    # step 4: train svm 
    print("-- Starting to Train SVM model --- ")
    svm_train.main(feature_file_name)


    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    run_pipeline()
