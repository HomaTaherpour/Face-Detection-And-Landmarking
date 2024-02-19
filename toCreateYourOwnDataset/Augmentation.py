import os
import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm

input_folder = "/Users/homa/Desktop/untitled folder/Face detection and landmarking Homa Taherpour/dataset/trainingset/Homa's image "
output_folder = "/Users/homa/Desktop/untitled folder/Face detection and landmarking Homa Taherpour/dataset/trainingset/augmented photos"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Define the augmentation sequence without flipping and rotating
augmentation = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(shear=(-16, 16), scale=(0.8, 1.2))),  # random shearing and scaling
    iaa.GaussianBlur(sigma=(0.0, 3.0)),  # random Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # random Gaussian noise
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # random brightness adjustment
    iaa.ContrastNormalization((0.8, 1.2)),  # random contrast adjustment
    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),  # elastic deformation
    iaa.Grayscale(alpha=(0.0, 1.0)),  # convert to grayscale with varying intensity
    iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.1, 0.1))),  # crop and pad
    iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),  # perspective transformation
    iaa.Sometimes(0.5, iaa.MultiplyHue((0.5, 1.5))),  # change hue
])

# Loop through each image and perform augmentation
for image_file in tqdm(image_files, desc="Augmenting Images"):
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Apply augmentation
    augmented_image = augmentation(image=image)

    # Save the augmented image
    output_path = os.path.join(output_folder, f"augmented_{image_file}")
    cv2.imwrite(output_path, augmented_image)

print("Augmentation complete.")
