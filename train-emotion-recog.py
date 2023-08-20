from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

#rescale images, initialize image data generator
train_data = ImageDataGenerator(rescale=1./255)
validation_data = ImageDataGenerator(rescale=1./255)

#preprocess train images
trained_gen = train_data.flow_from_directory(
    'train',
    target_size = (48,48),
    batch_size =128,
    color_mode = "grayscale",
    class_mode = "categorical"
) #flow_through_directory flows through directories, collects data, preprocesses

#preprocess test images
validation_gen = validation_data.flow_from_directory(
    'test', #target folder
    target_size = (48,48),
    batch_size =32,
    color_mode = "grayscale",
    class_mode = "categorical" #categories created according to the folders
)

emotion_recognition = Sequential()

emotion_recognition.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(48,48,1)))
emotion_recognition.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
emotion_recognition.add(MaxPooling2D(pool_size=(2,2)))
emotion_recognition.add(Dropout(0.25))

emotion_recognition.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
emotion_recognition.add(MaxPooling2D(pool_size=(2,2)))
emotion_recognition.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
emotion_recognition.add(MaxPooling2D(pool_size=(2,2)))
emotion_recognition.add(Dropout(0.25))

emotion_recognition.add(Flatten())
emotion_recognition.add(Dense(1024, activation="relu"))
emotion_recognition.add(Dropout(0.25))
emotion_recognition.add(Dense(7, activation="softmax"))

#compile convolutional layer with loss function optimizer- Adam
emotion_recognition.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#train neural network
model_info = emotion_recognition.fit_generator(
    trained_gen,
    steps_per_epoch=28709//32,
    validation_data = validation_gen,
    validation_steps = 7178//32
)

#save to json
mod_json = emotion_recognition.to_json()
with open("emotion_recognition.json", "w") as json_file:
    json_file.write(mod_json)

#save trained model file
emotion_recognition.save_weights('emotion_recognition.h5')