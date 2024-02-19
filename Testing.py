import os
import cv2
import numpy as np
import tensorflow as tf


WINDOW_NAME = 'Facial Landmarks Detection With Homa'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

def openWebcam(selected_landmark_indices):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    np.random.seed(42)
    tf.random.set_seed(42)

    image_h = 512
    image_w = 512

    model_path = os.path.join("keepingModelAndData", "LandMarkingModel.h5")
    model = tf.keras.models.load_model(model_path)

    all_landmarks = list(range(0, 136))  # Assuming there are 68 landmarks (2 * 68 = 136)

    use_selected_landmarks = True

    while True:
        ret, frame = cap.read()
        frame_with_detection, faces = face_detection(frame)
        frame_resized = cv2.resize(frame_with_detection, (image_w, image_h))
        frame_resized = frame_resized / 255.0

        input_frame = frame_resized[np.newaxis, ...].astype(np.float32)

        if len(faces) > 0:
            predictions = model.predict(input_frame, verbose=0)

            if use_selected_landmarks:
                frame_with_landmarks = plot_selected_landmarks(frame_with_detection.copy(), predictions[0], selected_landmark_indices)
            else:
                frame_with_landmarks = plot_selected_landmarks(frame_with_detection.copy(), predictions[0], all_landmarks)

            cv2.imshow(WINDOW_NAME, frame_with_landmarks)
        else:
            cv2.imshow(WINDOW_NAME, frame_with_detection)

        key = cv2.waitKey(1)

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('a') or key == ord('A'):
            selected_landmark_indices = []
            # Populate selected_landmark_indices with elements from 1 to 105
            for i in range(1, 106):
                selected_landmark_indices.append(i)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def plot_selected_landmarks(image, landmarks, selected_landmark_indices):
    h, w, _ = image.shape
    radius = 5

    for index in selected_landmark_indices:
        x = int(landmarks[index * 2] * w)
        y = int(landmarks[index * 2 + 1] * h)

        image = cv2.circle(image, (x, y), radius, (0, 255, 0), -1)

    return image

def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return frame, faces

if __name__ == "__main__":
    # Read the list from the .txt file
    with open("chosen_points.txt", "r") as file:
        selected_landmark_indices = eval(file.read())

    openWebcam(selected_landmark_indices)
