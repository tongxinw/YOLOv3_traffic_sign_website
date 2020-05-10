---
layout: post
title: Initial Blog Post
subtitle: First stage of our project 
bigimg: /img/yologo.png
gh-repo: https://github.com/tongxinw/YOLOv3_traffic_sign
gh-badge: [star, fork, follow]
tags: traffic_sign yolov3 object_detection
comments: true
---

# Introduction to object detection and YOLO

Ten years ago, it was nearly impossible for researchers to let computers tell the difference between cat and dog. Today, with the advanced development in image classification and object detection allow the computer to tell the difference with 99% accuracy. Object detection is a computer technology that combines image processing and computer visions to detect objects of a certain class, such as humans, cars etc. In current society, it is widely used in tracking objects, including video surveillance and image retrieval. Among various methods for object detection, YOLO (You Only Look Once) utilized Convolutional Neural Network (CNN) to perform end-to-end object detection without defining features.
The following diagram illustrates the architecture of the CNN used in YOLOv3.

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/1400/0*QW4v12jc29S6fmAt" alt="Architecture of the CNN">
</div>
<br/>

The Method was first proposed by Joseph Redmon et al. from University of Washington in 2015 and has been updated to version 3 in 2018 along with another researcher Ali Farhadi in the paper titled “YOLOv3: An Incremental Improvement”.

Past advanced detection systems such as R-CNN employ region proposal methods. Given an image, such systems first generate potential bounding boxes and then run a classifier on the proposed boxes. Post-processing is used after classification to refine bounding boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene. Such complex pipelines are slow and hard to optimize since each individual component needs to be trained separately.

YOLO is a unified detection system. It is based on a single convolutional network Thus, YOLO is more efficient compared to other detection systems. YOLO reasons globally about an image, and thus makes less background errors, in contrast to region proposal-based techniques.

The approach applies a single neural network trained end to end to the full image. “This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities” (Redmon).

For example, the image shown below can be divided into a S * S grid and each cell in the grid are assigned with corresponding class probability map and the system also predicts bounding boxes using dimension clusters and predicts an objectness score for each bounding box using logistic regression. As the result, the class probability map and the bounding boxes with confidences are combined and generate a final detection of bounding boxes and class labels.

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/1400/1*8eGPJMRdeHxxFKV6grSbpw.png" alt="Example">
</div>
<br/>

YOLOv3 outperformed former versions with its extremely fast speed and high performance under the help of algorithms such as multilabel classification as well as independent logistic classifiers.

# Code implementation and explanation

We started our project from the official [DarkNet GitHub repository](https://github.com/pjreddie/darknet), coming with the paper, “[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)”. The official github contains the source code for the YOLOv3 implemented in the paper (written in C), providing a step-by-step tutorial on how to use the code for object detection.

It is a challenging task to transfer the coding implemented in C to Keras in Python . For example, even using a pre-trained model directly requires sophisticated code to distill and interpret the predicted bounding boxes output by the model. As a result, we learned Keras implementation from the a great Github post, “[keras-yolo3: Training and Detecting Objects with YOLO3](https://github.com/experiencor/keras-yolo3)” by [Huynh Ngoc Anh](https://www.linkedin.com/in/ngoca/).

The first step is to download the pre-trained model weights.

These were trained using the DarkNet code base on the MSCOCO dataset. Download the model weights and place them into the current working directory with the filename “*yolov3.weights.*”

Next, we need to define a Keras model that has the right number and type of layers to match the downloaded model weights. The model architecture is called a “*DarkNet*” and was originally loosely based on the VGG-16 model. Generally, the YOLOv3 is structured as the following:

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/1064/1*KFtyQ2qSchYegpc9tWnCJQ.png" alt="yolov3 structure">
</div>
<br/>

Specifically, the following figure displays a shortcut of the YOLOv3 model that we used for our project:

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/932/1*Y_jSVCd0q4sCskBPXKidfw.png" alt="yolov3 model">
</div>
<br/>

After defining the model and downloading the pre-trained weights, we call the *load_weights()* function to pass the weights into the model and set up the weights to specified layers.

Then, we saved the model for further predictions.

Since the model was pre-trained with dedicated classes, the model we used can only detect classes listed below:

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/1400/1*W0NSAZCU-AocR-ZwNHdoWQ.png" alt="80classes">
</div>
<br/>

Finally, we will give some instances detected by the model. The input test images should be loaded, resized and scaled to the suitable format for detecting, which are expected to be color images with the square shape of 416*416 pixels scaling from 0–1 in this case. However, the output of the model is encoded bounding boxes and class predictions, which needs further interpretation. Thus we draw the bounding boxes on the original images to do the visualization.

The following cases are the examples running the YOLOv3 model:

1. YOLOv3 detects a single person in the image with a high accuracy, which is over 97%.

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/840/1*6fwuESAW2eK6GDQn3bip5Q.png" alt="test_person">
</div>
<br/>

2. When the image contains more than one object, our selected YOLOv3 model could also detect those objects one by one. Since YOLOv3 sees the entire image while prediction, we can see that there are few background errors in the following instance, which is one of the strengths of the YOLOv3 model compared to other object detection algorithms. However, it evokes one limitation of the YOLOv3 model. When multiple objects gather together, it is possible for the YOLOv3 model to generate lower accuracy for the object detection.

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/1400/1*5_KgxbyOfJpD0_bPrUNAsg.png" alt="test_person2">
</div>
<br/>

3. Another limitation of the YOLOv3 model is represented by the following images. It struggles to localize small objects that appear in groups. We can see from the following two instances that it fails to detect some of the people, and for the flock of birds, it may confuse the YOLOv3 model which loses the ability to detect them separately.

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/864/1*lWt0mJ7ZVyAFtc6lpoZ2Eg.png" alt="test_person3">
</div>
<br/>

<div style="text-align:center;">
  <img src="https://miro.medium.com/max/1400/1*V83j6qZ4YdotJD32r4mB9A.png" alt="test_bird">
</div>
<br/>

# Next Steps

With the pretrained model using YOLOv3 which could detect over 80 categories, we want to extend the model by training with our custom dataset. In the next stage, we will focus on the detection of traffic signs, which are key map features for navigation, traffic control and road safety. In the bright future of autonomous driving, accurate and robust detection of traffic signs is a crucial step for driving directions and early warning.

Our training and test dataset come from one of Google’s open source, OpenImageV6, which is a public database online. It contains a total of 16M bounding boxes for 600 object classes on 1.9M images, making it the largest existing dataset with object location annotations. The boxes have been largely manually drawn by professional annotators to ensure accuracy and consistency. The images are very diverse and often contain complex scenes with several objects (8.3 per image on average). In this case, we will make the use of only one of the categories, traffic signs, to retrain our model. The images and labels are downloaded into separate folders.

Here is the detail instruction to download the dataset from OpenImageV6: [Colab Coding Instruction](https://colab.research.google.com/drive/1oJ8tI2ghtj7U0gc67Fl_HltzQYarfix1)

# Reference
[Ayoosh Kathuria, What’s new in YOLO v3? Towards Data Science.](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

[Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

[Joseph Redmon & Ali Farhadi, YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)


