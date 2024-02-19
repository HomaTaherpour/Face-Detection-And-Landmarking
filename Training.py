import cv2 # we use cv2 for resize and read the images ,face detection and opening the webcam
import tensorflow
from glob import glob # Usage in loading the dataset path matching pattern
import numpy # images are array so we need  numpy
import os

global imageHeight
global imageWidth
global landmarksNumbers
# error handling for tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def pathChecker(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    # loading whatever end with .jpg and .text
    trainingImage = sorted(glob(os.path.join(path, "trainingset", "images", "*.jpg")))
    trainingLandmark = sorted(glob(os.path.join(path, "trainingset", "landmarks", "*.txt")))
    validationImage = sorted(glob(os.path.join(path, "validationset", "images", "*.jpg")))
    validationLandmark = sorted(glob(os.path.join(path, "validationset", "landmarks", "*.txt")))
    return (trainingImage, trainingLandmark), (validationImage, validationLandmark)

def imageLankmarksINFO(image_path, landmark_path):
    photo = cv2.imread(image_path, cv2.IMREAD_COLOR)
    Height, Width, _ = photo.shape #get the actual size to decide how much increase or decrease the size of each
    photo = cv2.resize(photo, (imageWidth, imageHeight))# resizing

    photo = photo/255.0# normalizing the image
    photo = photo.astype(numpy.float32)

    #Lankmarks
    data = open(landmark_path, "r").read()# getting the landmarks from the .text in files and put in data variables
    photoslankmarks = []# save the landmark information

    for line in data.strip().split("\n")[1:]:# the first line is the number of each landmark in each .txt (106)
        x, y = line.split(" ")
        x = float(x)/Height
        y = float(y)/Width # by deviding this two to Height and Width we are normlizing it to a number bettween 0 and 1 for each photo

        photoslankmarks.append(x)
        photoslankmarks.append(y)

    photoslankmarks = numpy.array(photoslankmarks, dtype=numpy.float32) # converting the list to an array and converting to float

    return photo, photoslankmarks

def preprocessTheImage(imagePath, LandmarkPath):# X imagePath #Y LandmarkPath
    def decoder(imagePath, LandmarkPath):
        imagePath = imagePath.decode() # file are coded so we need to decode it
        LandmarkPath = LandmarkPath.decode()

        image, landmarks = imageLankmarksINFO(imagePath, LandmarkPath)
        return image, landmarks
    # give the value to tensorflow to that get executed in the form of the variables data type
    image, landmarks = tensorflow.numpy_function(decoder, [imagePath, LandmarkPath], [tensorflow.float32, tensorflow.float32])
    image.set_shape([imageHeight, imageWidth, 3])
    landmarks.set_shape([landmarksNumbers * 2])

    return image, landmarks


def tensorflowDatasetChanger(listOfImagesFilePath, listOfLandmarkFilePath, batch=8):
    ds = tensorflow.data.Dataset.from_tensor_slices((listOfImagesFilePath, listOfLandmarkFilePath))
    ds = ds.shuffle(buffer_size=5000).map(preprocessTheImage) # shuffel the dataset and convert them to batches, for optimization
    ds = ds.batch(batch).prefetch(2)
    return ds

def build_model(input_shape, num_landmarks):
    inputs = tensorflow.keras.layers.Input(input_shape)
    backbone = tensorflow.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=0.5)# alpha is for increase and decrease the number of filters
    backbone.trainable = True # set the backbone as trainable

    x = backbone.output
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    x = tensorflow.keras.layers.Dropout(0.2)(x)
    outputs = tensorflow.keras.layers.Dense(num_landmarks*2, activation="sigmoid")(x)# since the landmarks are in the range of 0 and 1 we use sigmoid
    model = tensorflow.keras.models.Model(inputs, outputs)
    return model

if __name__ == "__main__":

    numpy.random.seed(42)
    tensorflow.random.set_seed(42)


    pathChecker("keepingModelAndData2")

    #Hyperparameters
    imageHeight = 512 #input size is imageHeight*imageWidth*3
    imageWidth = 512
    landmarksNumbers = 106
    input_shape = (imageHeight, imageWidth, 3)# 3 is pointing to RGB images
    batch_size = 32
    learningRate = 1e-3
    epochsNumbers = 100


    dataset_path = "/Users/homa/Desktop/Face detection and landmarking Homa Taherpour/dataset"
    model_path = os.path.join("keepingModelAndData2", "LandMarkingModel.h5")
    csv_path = os.path.join("keepingModelAndData2", "data.csv")

    """ Loading the dataset """
    (trainingImage, trainingLandmark), (validationImage, validationLandmark) = load_dataset(dataset_path)
    print(f"Training pictures and landmark info: {len(trainingImage)}/{len(trainingLandmark)} - Validpictures and landmark info: {len(validationImage)}/{len(validationLandmark)} ")
    print("")

    train_ds = tensorflowDatasetChanger(trainingImage, trainingLandmark, batch=batch_size)
    valid_ds = tensorflowDatasetChanger(validationImage, validationLandmark, batch=batch_size)

    model = build_model(input_shape, landmarksNumbers)
    model.compile(loss="binary_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(learningRate))

    callbacks = [
        # save the weight file during the train
        tensorflow.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        # if the val loss function didn't decrease during 5 epoch decrease the learning rate
        tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        # add the information about loss etc. to csv
        tensorflow.keras.callbacks.CSVLogger(csv_path, append=True),
        # if val loss didn't decrease training will stop
        tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_ds,validation_data=valid_ds,epochs=epochsNumbers,callbacks=callbacks)

