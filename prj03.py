import cv2
import os


target_width = 275
target_height = 275
input_root = 'dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/asl_processed/train'
output_folder = 'dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/prj03_train'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(root, file)
            input_img = cv2.imread(img_path)

            if input_img is None:
                continue
            height, width, channels = input_img.shape
            
            if height > target_height or width > target_width:
                input_img = cv2.resize(input_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                height, width, _ = input_img.shape
            
            #calc padding size 
            pad_wid = target_width - width
            left_pad = pad_wid // 2
            right_pad = pad_wid - left_pad
            pad_hgt = target_height - height
            top_pad = pad_hgt // 2
            bottom_pad = pad_hgt - top_pad

            #padding, blur, normalization
            res_img = cv2.copyMakeBorder(input_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            blurred_img = cv2.GaussianBlur(res_img, (5, 5), 0)
            norm_img = blurred_img.astype("float32") / 255.0                
            final_img = (norm_img * 255).astype("uint8")
            out_path = os.path.join(output_folder, file)
            cv2.imwrite(out_path, final_img)

