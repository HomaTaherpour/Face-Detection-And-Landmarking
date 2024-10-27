# Face Detection And Landmarking


<img width="866" alt="Face-Detection-And-Landmarking" src="https://github.com/user-attachments/assets/ce9b4985-d876-498a-8f35-be574ca5cdad">

![Face-Detection-And-Landmarking](https://github.com/user-attachments/assets/18a27235-22f4-45f2-8bfe-e10175a84ddd)

## Introduction

This repository contains scripts tailored for facial landmark detection, offering features like real-time testing, a graphical user interface for landmark selection, image augmentation, and pixel coordinate extraction. The code efficiently employs MobileNetV2 for facial landmark detection, ensuring a balance between model size and accuracy. The model architecture includes a Global Average Pooling layer, dropout, and a Dense layer with sigmoid activation. Noteworthy scripts such as agumantation.py and dimentationtaker.py enable customization based on your images and dataset. Utilize dimentationtaker.py to overlay radio buttons on images and agumantation.py to augment pictures for dataset creation.

## Prerequisites
- Opencv
- TensorFlow
- Numpy
- Imgaug
- tqdm
- PIL (Python Imaging Library)
- Tkinter (for GUI)


## dataset 
For training and evaluating the model, consider utilizing the lapa dataset, accessible at:
[https://github.com/jd-opensource/lapa-dataset]


## License
This project is licensed under the [MIT License](LICENSE).
