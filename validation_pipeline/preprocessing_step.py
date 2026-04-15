import cv2
import os


target_width = 275
target_height = 275

#full dataset
input_root = '../dataset/ASL_HG_36000/asl_processed/test'
output_folder = 'validation'

#sample dataset
#input_root = 'prj03_sample_train'
#output_folder = 'segmentation/sample_out'

if not os.path.exists(input_root):
    print(f"Error: The path '{input_root}' does not exist!")
else:
    print(f"Path found! Searching for files in: {os.path.abspath(input_root)}")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(root, file)
            input_img = cv2.imread(img_path)

            if input_img is None:
                continue

            final_img = cv2.resize(input_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(output_folder, file)
            cv2.imwrite(out_path, final_img)

