# Vehicle Detection Project


---



The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/v_and_nonv.png
[image2]: ./output_images/vhog_nvhog.png
[image3]: ./output_images/all_rects.png
[image4]: ./output_images/6_processed_imgs.png
[image5]: ./output_images/one_zone_rects.png
[multizone_rects]: ./output_images/multizone_rects.png
[heatmap]: ./output_images/heatmap.png
[filtered_heatmap]: ./output_images/filtered_heatmap.png
[image6]: ./output_images/gray_heatmap.png
[image7]: ./output_images/post_heatmap_boxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the "Visualize Images" code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In cell one, under the sub-heading HOG Feature Extraction, I used the function `get_hog_features()` to calculate and visualize the orientation of gradients in the various images. It's not obvious at first, but the vehicles tend to have variety of lines and edges, causing the visualization of gradients to almost resemble fireworks.  Non-vehicles tend to have more random HOG features. Here are examples of how the gradient directions are similar with vehicles, and more random with non-vehicles:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters through trial and error.  I briefly attempted to set up function that would run through a variety of SVM parameters and display them, but my computer is too slow.  Here are the final HOG parameters:

| HOG Parameter   |  Value       |  
| -------------   |-------------:|
| Color Space     | "YCrCb" |
| Orientations    | 12     |   
| Pixels Per Cell | 16      |    
| Cells Per Block | 2     |
| HOG Channels    | "ALL" |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features.

I trained a `LinearSVC()` using Scikit Learn's linear SVM library.  Trained just on the HOG features, it achieved a .9868 accuracy and took 0.00316 seconds to predict 10 labels (all correct) after training.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding windows was another trial and error process. The difficulty I faced most was finding a balance between having too many false positives, and not picking up the cars if they got far enough away.

I remedied this by having multiple small scale search zones in the distance - the smaller `ystart` and `ystop` values - and fewer large search zones in the foreground.  Here is a visualization of all the search zones:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on six scales using all YCrCb channel HOG features.
This created ok identifications with a few false positives:

![alt tx][image4]

Most of these false positives were fixed when I started keeping track of rectangles across multiple video frames.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  

![alt text][multizone_rects]

From the positive detections I created a heatmap:

![alt text][heatmap]

And then thresholded that map to identify vehicle positions:  

![alt text][filtered_heatmap]

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap:

![alt text][image6]

I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Vehicles near each other tend to be classified as one object instead of two.  

* The pipeline will probably fail if car looks significantly different from vehicles in the training set.  For instance, a motorcycle is technically a vehicle, but it is different looking enough where my classifier may have difficulty recognize one.  

* If a car was traveling with something tied to it's roof, or bicycles on the back of it the classifier may have difficulty.  If it's towing something that doesn't look like a vehicle.

* There are some false positives when the sliding windows interpret the guard rails or trees as vehicles.  This is not horrible, because you would want your car to avoid guard rails and trees, but they obviously aren't vehicles.  

* Distant cars get lost once briefly, which makes sense because they take up fewer pixels and would have less detailed gradients.  Again, in practice, it may not be overly necessary to recognize objects that are not close enough to cause a collision.

### Future Improvements:

* Train on multiple types of vehicles (compact, pick up, motorcycle, big rig, etc.), so the classifier doesn't have such a wide number to shapes to identify in the large category that is "Vehicles."

* I had difficulty getting my sliding windows all the way to the right edge of the images.  That is something I'd like to improve when not under the time constraints of the end of Term 1.

* Including xstart and xstop parameters might help `potential_vehicles()` have fewer false positives on the side of the road.
