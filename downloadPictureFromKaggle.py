'''USE THIS LIKE TO FIND THE DATASET : https://www.kaggle.com/c/facial-keypoints-detection/leaderboard'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load CSV file into a Pandas DataFrame
csv_file_path = '/Users/homa/Desktop/Face detection and landmarking Homa Taherpour/second/face landmark detection/dataset/facial-keypoints-detection/training.csv'
df = pd.read_csv(csv_file_path)

# Convert the image data from string to numpy array
df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' ', dtype=int) if isinstance(x, str) else None)

# Create the output folder if it doesn't exist
output_folder = '/Users/homa/Desktop/Face detection and landmarking Homa Taherpour/second/face landmark detection/dataset/facial-keypoints-detection/untitled folder'
os.makedirs(output_folder, exist_ok=True)

# Create images from the pixel data and visualize them
for index, row in df.iterrows():
    # Check if the 'Image' column contains a valid array
    if row['Image'] is not None and isinstance(row['Image'], np.ndarray):
        # Determine compatible shape based on the array size
        size = len(row['Image'])
        sqrt_size = int(size**0.5)

        # Find the closest integers whose product is equal to or greater than the array size
        for i in range(sqrt_size, 0, -1):
            if size % i == 0:
                compatible_shape = (i, size // i)
                break

        # Reshape the 1D array to a 2D array with a compatible shape
        image_array = row['Image'].reshape(compatible_shape)

        # Plot the image
        plt.imshow(image_array, cmap='gray')

        # Check if keypoints are available
        if 'left_eye_center_x' in row.index:
            # Plot facial keypoints if available
            keypoints = row.drop('Image').values
            keypoints = keypoints.astype(float).reshape(-1, 2)
            plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', marker='o', s=10)

        # Save or display the plot
        image_path = f'{output_folder}/image_{index}.png'
        plt.savefig(image_path)
        plt.close()
    else:
        print(f"Skipping row {index} due to None or invalid 'Image' value.")

print("Images with keypoints saved successfully.")
