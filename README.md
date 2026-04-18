# Project 03 Report:
## Preprocessing:
The dataset was already preprocessed including MediaPipe hand segmented cropping. Some of the images were different sizes because of the extension of fingers in certain characters, so we ran a script to explore the sizing. Testing with 1 image per sample class, we found the smallest image was 100x103, the largest was 217x275, and the average size was 147x20.

<img width="580" height="262" alt="image6" src="https://github.com/user-attachments/assets/0b63d93b-0480-422d-a85f-f94eb7b0e279" />


Our original preprocessing consisted of adding padding, creating fixed sizing, applying a Gaussian blur, and normalizing the image.

<img width="541" height="285" alt="image5" src="https://github.com/user-attachments/assets/744bf7a1-a5db-4708-a602-9b2f00d0599a" />

  
The padding created an artificial background that can confuse the model when the input does not match the training set. The purpose of the Gaussian blur was to reduce noise in the image for edge detection, but it resulted in removing valuable details in the image. Ultimately, we chose to just resize all of the images to 275x275 to match the largest image for fixed model input sizes. 

<img width="563" height="248" alt="image3" src="https://github.com/user-attachments/assets/2f46c83c-2c80-4604-bbc5-f9e015e2a593" />



## Segmentation:
Our next step in the pipeline was segmentation, with the goal of isolating the hand region from the background. The dataset contains variation in skin tone, lighting, and background complexity, which introduces noise and makes segmentation challenging. To address this, we adopted a method that relies on intensity and structural features rather than strict color-based thresholds.

<ins> Rejected Alternative Methods </ins> 

We rejected two alternative methods. 
1. Chan-Vese: While effective for smooth object boundaries, this method produced inconsistent results for hand images with complex finger configurations. It struggled to accurately capture separated or bent fingers, often merging or missing fine structures.
2. Watershed: This method frequently over-segmented the hand, introducing artificial boundaries within the hand region. Its sensitivity to small intensity variations resulted in fragmented and noisy masks.

<img width="1999" height="568" alt="image8" src="https://github.com/user-attachments/assets/babb2123-3299-444e-acc1-e042f181f989" />


<ins> Final Segmentation Method </ins>

Our final approach uses color-based skin segmentation combined with morphological refinement.
Each image is checked to ensure it is sized 275×275 and converted into two different color spaces: HSV and YCrCb. These color spaces are commonly used for skin detection because they separate luminance (brightness) from chrominance (color), making them more robust to lighting variations.
We apply predefined threshold ranges in both color spaces to identify pixels likely to correspond to skin:

* In HSV, thresholds capture typical skin hue and saturation ranges
* In YCrCb, thresholds isolate skin tones based on chrominance channels (Cr and Cb)
  
The resulting masks from both color spaces are combined using a logical AND operation. This intersection helps reduce false positives by only keeping pixels that satisfy both skin color models.
To refine the segmentation, several morphological operations are applied:
* Closing (with a disk kernel): fills small gaps and connects nearby regions
* Hole filling: ensures the hand region is solid
* Removal of small objects: eliminates noise and small background artifacts
* Removal of small holes: smooths the interior of the hand region

Finally, the refined binary mask is applied to the original image to extract the segmented hand.
This approach proved to be more robust than Chan-Vese and Watershed because it directly leverages skin color information while also incorporating spatial refinement. By combining multiple color spaces and morphological processing, the method effectively handles variations in lighting, background noise, and hand shape.

![image2](https://github.com/user-attachments/assets/ae692012-19bf-4374-9a3b-0a4021d5f47c)
![image4](https://github.com/user-attachments/assets/30170442-9a6f-470d-a74a-311af2bc4b4b)


## Feature Extraction:
After obtaining the segmented hand images using color-based skin detection, we extract features that capture both the overall hand shape and the fine structural details of the fingers. Our method combines mask-based features and edge-based features, producing a robust representation suitable for classification.

<img width="437" height="443" alt="image7" src="https://github.com/user-attachments/assets/898f9e2a-22d5-489d-9c2e-94f5f8517846" />


<ins> Mask Features: </ins>

The binary mask from segmentation identifies the hand region. This captures the global geometry of the hand, including the overall shape and finger positions. By representing which pixels belong to the hand, the mask provides essential information about hand posture and size.

<ins> Edge Features: </ins>

To capture finer details, we apply a Gaussian filter to smooth the image and reduce noise, followed by Canny edge detection to highlight strong intensity gradients corresponding to finger boundaries and hand contours. Edge features emphasize local structural differences, which are critical for distinguishing gestures that differ by subtle finger positions.

<ins> Dimensionality Reduction: </ins>

Both mask and edge maps are downsampled from 275×275 to 64×64. This reduces the feature dimensionality, improves computational efficiency, and ensures that our model can process approximately 30,000 images without excessive memory or runtime requirements.

<ins>Feature Vector Construction:</ins>

The downsampled mask and edge maps are flattened and concatenated into a single vector. Edge features are given a higher weight, emphasizing fine structural differences while still preserving the global hand shape.

<ins> Normalization: </ins>

The final feature vectors are standardized to zero mean and unit variance. This step ensures that all features contribute equally to the classifier, preventing features with larger values from dominating the model and improving SVM performance.
This hybrid representation effectively balances global hand geometry and local finger details, making it robust to variations in hand size, shape, and orientation while remaining discriminative for gesture classification. Preliminary tests using an SVM classifier demonstrate that this feature extraction method captures meaningful information and is well-suited for distinguishing between visually similar hand gestures.

<img width="1999" height="1714" alt="image1" src="https://github.com/user-attachments/assets/4e7436f5-7e60-4da9-af7a-1d6126652a05" />


## Instructions:
* Download repo as zip file on main branch
* Update paths for sample data in prj03.py and run prj03.py
  * Comment out the path to full dataset (lines 9 and 10) and use commented out sample paths (lines 13 and 14)
* Go to segmentation directory and update path on segmentation.py on line 12 with directory with images to test (sample_out)
  * scikit-image version must be 0.26 OR switch line 37 from max_size arg to min_size and 38 from max_size to area_threshold
* Run the images on feature_extraction.py



## Team Contributions:
* Amaya - preprocessing
* Genesis - segementation and feature extraction

# Project 04 Report:
## Classification Justification:
We selected a support vector machine (SVM) with a radial basis function (RBF) kernel because our feature space is high dimensional and contains non-linearly separable classes. Many ASL hand gestures are very similar, with the only distinguishable difference being a tuck of a finger. For example, the character m is represented by making a fist and tucking your tumb between your pinky and your index finger an the character n is represented by tucking your thumb between your index finger and your middle finger. 
<img width="517" height="523" alt="Screenshot 2026-04-17 at 10 30 45 PM" src="https://github.com/user-attachments/assets/18a46972-0f86-4bce-8b9f-c82e33ed2263" />
<img width="524" height="523" alt="Screenshot 2026-04-17 at 10 32 03 PM" src="https://github.com/user-attachments/assets/76bc1110-fdaa-4d90-b780-5dc152d33a00" />

These similar hang gestures will produce similar masks that will be very close together in the feature space making them difficult to separate. Changing the kernel to radial basis function projects the input to higher dimensions to distinguish between non linearly separable hand gestures and improve classification. 

## Validation Results:
When we ran our validation pipeline, our resulting accuracy was 99.94%. We thought this was suspiciously high so we ran the pipeline on some images of our own and we got a validation accuracy of 10%. When we ran the pipeline on the sample of images we provided you our accuracy was 100%. We thought there was an issue with segmentation, but multiple updates did not solve the problem. There is most likely an overfitting issue somewhere, but we are unsure where. A possible issue could be that our model isn’t necessarily learning the different signs, but instead memorizing the backgrounds. 

* Results of the validation pipeline are available here:  https://docs.google.com/document/d/1y8do3e7dDvVR8SrR_F1-IYwb2Ai_aprwYx_s_SJlH6Y/edit?usp=sharing

## Improvements
We believe that we need to either figure out if the issue is in segmentation or feature extraction (or maybe both). Once we can figure that out, we could look into different ways of implementing these algorithms. We can figure out where the issue is by looking at the segmentation maps and feature maps. 

Another way we can try to improve our model is to perform data augmentation on our training data when we train the svm model so it isn’t learning just the background. However, we have to be careful to not change the meaning in our signs. 

 
## Instructions for running:
* Download validation sample images below and place in validation_pipeline folder
  * https://drive.google.com/drive/folders/1sjwdfVyEQHp7ftuBRSvnoKHXj2UDgoa5?usp=sharing 
* Download SVM model below and place in validation_pipeline folder
  * https://drive.google.com/file/d/1AdlnGHOqG_RFAGPw8RNGra9ollUklk27/view?usp=drive_link 
* Run validation_pipeline/segmentation.py 
* Run validation_pipeline/feature_extraction.py
* Run validation_pipeline/svm_test.py

## Team Contributions:
* Gen: trained SVM
* Amaya: wrote report
















