Colab:   
* Full code w/ training: [https://colab.research.google.com/drive/1TymQCHYQOcqySuFOSeVN928u9JePMBIp?usp=sharing](https://colab.research.google.com/drive/1TymQCHYQOcqySuFOSeVN928u9JePMBIp?usp=sharing) 

* Code to test:   
[https://colab.research.google.com/drive/1trcTQrfr0wV01F5MzVOHKR4j0cV9ZVzE?usp=sharing](https://colab.research.google.com/drive/1trcTQrfr0wV01F5MzVOHKR4j0cV9ZVzE?usp=sharing) 

* Video Link:  
[vid link](https://drive.google.com/file/d/1tFfwXf5IR93jUU159rDbWCNMeEoZwYJm/view?usp=sharing)

The custom test database developed for this project consists of 350 total images, spanning 35 classes that include the letters A–Z and the numbers 1–9. Each class contains 10 unique samples captured from Genesis and Amaya against a consistent, clean white background with uniform lighting. This controlled environment marks a significant departure from the training and validation subsets, which relied on more "noisy" data featuring varied backgrounds and fluctuating light conditions. However, the complexity of this test set lies in its structural variance rather than its environment. While the training data predominantly featured front-facing palms, 
<p align="center">
  <img src="https://github.com/user-attachments/assets/c2d67f99-4b43-4f5d-bf63-7550016b0409" width="188" alt="Amaya’s C" />
  <img src="https://github.com/user-attachments/assets/97541339-c416-4319-a61c-74e0d89518fb" width="183" alt="Genesis’s C" />
  <img src="https://github.com/user-attachments/assets/3b43b2c1-1a9d-495b-a81c-6a0ab878e59d" width="189" alt="Training C" />
  <br>
  <b>Figure 1: Amaya’s C, Genesis’s C, and Training C.</b>
  <p>This dataset introduces significant changes in hand orientation, including lateral angles, rotations, and even reflected or "mirrored" versions of the original signs.</p>
</p>


These structural differences are sufficient to test the robustness of the final project because they force the model to demonstrate true feature recognition rather than simple pattern matching. By eliminating background noise, we can isolate the model's ability to handle geometric transformations. The inclusion of reflections and varied orientations is a critical test of the model’s real-world utility; it determines whether the system can accurately interpret signs captured from a live camera feed where the user’s hand may be tilted or positioned at an unconventional angle. In ASL, people choose which hand is their base/dominant hand so it is important for our model to be able to recognize signs from either hand.  Ultimately, verifying that the model can recognize a sign despite these perspectival shifts is a more meaningful measure of robustness than simply testing its ability to filter out a cluttered background.

A quick recap about our training and validation accuracy; our training accuracy was 99.84% where we tested with 5780 images.    
<p align="center">
   <img src="https://github.com/user-attachments/assets/25e518aa-893b-409a-85bd-50b64fc27a0a" width="288" alt="Amaya’s C" />
  <br>
  <b>Figure 2: Training Confusion Matrix</b>
</p>


And our validation accuracy was 31.37%, where we tested on 180 images.   
<p align="center">
   <img src="https://github.com/user-attachments/assets/264990c1-9f29-4f9a-ad71-33607f493edb" width="288"/>
  <br>
  <b>Figure 3: Validation Confusion Matrix</b>
</p>

We had to change our segmentation process during this sprint because when we tested our previous implementation, it had a close to 0% accuracy. Clearly our model just overfitted to possibly the background so we tried a few other things. First, we tried using DINO with the hand query and if that failed, the fallback option was to use SAM. If SAM failed, we decided to just keep the original image and then go to feature extraction (HOG). We decided to keep the original images even if they weren’t segmented because having “noisy” images is better than having no images for a certain class at all. Also, we used a different dataset for our validation ([https://www.kaggle.com/datasets/ayuraj/asl-dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset))  to ensure that our model wasn’t just overfitting to our training dataset. Thus, this could explain the drop in accuracy for our validation test. 

Now, our accuracy for our test dataset was 14.57% (better than random chance :) ). The big gaps between training, validation, and testing accuracies could be explained by our not so good segmentation process. Our segmentation was cutting off images and fingers in training data. In our validation set, the images were already close to segmented so it was much easier for our pipeline and the images were taken from a further distance. The drop in our test set could be due to the different angles/reflections on each class which was different and not really seen in the training and validation dataset.   
<p align="center">
   <img src="https://github.com/user-attachments/assets/9ddde050-bad6-4883-b44e-4185253d1b35" width="288" />
  <br>
  <b>Figure 4: Test dataset accuracy</b>
</p>

  <p align="center">
   <img src="https://github.com/user-attachments/assets/08a9f963-3288-4daf-a890-6e54281edf6e" width="388" />
  <br>
  <b>Figure 5: Best and Worst letters for each tests  </b>
</p>



Figure 5 displays the top and bottom  3 signs based on accuracy for the training, validation, and testing dataset. The signs shown are based on our segmentation and as you can see, our model removes fingers and gets confused on the spacing between fingers. 

The main improvement we need to make is to take more time on our segmentation pipeline and ensure that all of our images are being properly segmented. I think that we got overwhelmed with the noisy background and overcomplicated the segmentation which hurt us at the end. 

  <p align="center">
   <img src="https://github.com/user-attachments/assets/9c97f2c2-0bc2-4f7b-91cd-1821094a74f3"" width="388" />
  <br>
  <b>Figure 6: How bad our segmentation is</b>
</p>

 
Images from the same person and the same class produced varying masks, demonstrating the inconsistency of our segmentation.   
**Contributions**

Both: Fighting segmentation, creating testing dataset

Genesis: Report, scripts so everything is in one file

Amaya: Most of segmentation, moving to colab, running training, validation, and testing pipelines

**How to run:** 

- Open [dex\_final folder](https://drive.google.com/drive/folders/1pMOQmaQpmZMfBOFEycxkcII6Q1kwiKX_?usp=drive_link) containing test images and SVM model  
- Add shortcut to drive so files can be accessed after mounting drive

<img width="647" height="302" alt="Screenshot 2026-05-04 at 6 51 49 PM" src="https://github.com/user-attachments/assets/92da096c-f3cd-42ee-b1cc-6c7f30e2855b" />


- Run the [CV Project 05](https://colab.research.google.com/drive/1trcTQrfr0wV01F5MzVOHKR4j0cV9ZVzE?usp=sharing)   
- The last cell requires you to upload an image of your own and it will predict the class  
  - (Please do Y if you want a chance of this working)

