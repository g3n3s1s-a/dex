import cv2
import os


def preprocessing(input_root,output_folder):

  target_width = 275
  target_height = 275

  #full dataset
  #input_root = 'dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/asl_processed/train'
  #output_folder = 'dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/prj03_resized_train'

  #sample dataset
  #input_root = 'prj03_sample_train'
  #output_folder = 'segmentation/sample_out'

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

if __name__ == "__main__":
  input_dir = 'dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/asl_processed/train'
  output_dir ='dataset/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/prj03_resized_train'
  preprocessing(input_dir,output_dir)

