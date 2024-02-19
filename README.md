# Face Detection And Landmarking

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
