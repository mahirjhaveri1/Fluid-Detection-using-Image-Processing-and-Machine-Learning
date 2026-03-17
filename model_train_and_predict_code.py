import zipfile
import os
zip_path = "/content/main_cam_data1.zip"
extract_path = "/content/main_cam_data1/main_cam_data1"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extracted folders:", os.listdir(extract_path))



#  Bottle Fill Level Classification (0%, 25%, 50%, 75%, 100%)
#  Using CNN + Image Processing + TensorFlow Lite Conversion
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Configuration
DATASET_DIR = "/content/main_cam_data1/main_cam_data1"
IMG_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 16
LABEL_MAP = {
    "images_0": 0,
    "images_25": 1,
    "images_50": 2,
    "images_75": 3,
    "images_100": 4
}
CLASS_NAMES = {
    0: "0% filled",
    1: "25% filled",
    2: "50% filled",
    3: "75% filled",
    4: "100% filled"
}
# Image Preprocessing (with contours)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to HSV to isolate blue liquid
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Contour detection for possible future use
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)

    # Normalize
    return img.astype(np.float32) / 255.0
# Dataset Loading
images, labels = [], []
for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    label = LABEL_MAP.get(folder)
    if label is None:
        continue

    for file in os.listdir(folder_path):
        try:
            img = preprocess_image(os.path.join(folder_path, file))
            images.append(img)
            labels.append(label)
        except:
            pass

X = np.array(images, dtype=np.float32)
y = to_categorical(labels, num_classes=5)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save Model & Convert to TFLite
model.save("bottlefillmodel_v3.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("bottlefillmodel_v3.tflite", "wb") as f:
    f.write(tflite_model)

print("\n TFLite model saved as 'bottlefillmodel_v3.tflite'")



#  Bottle Fill Level Prediction from Image Upload

#  Using TFLite Model
from google.colab import files

def upload_and_predict():
    uploaded = files.upload()
    interpreter = tf.lite.Interpreter(model_path="bottlefillmodel_v3.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    for filename in uploaded.keys():
        print(f"\nFile received: {filename}")

        # Preprocess
        img = preprocess_image(filename)
        input_data = np.expand_dims(img, axis=0)

        # Prediction
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        predicted_class = np.argmax(output_data)
        confidence = np.max(output_data)
        percentage = [0, 25, 50, 75, 100][predicted_class]

        print(f"Prediction: {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}% confidence)")

        # Soda Bottle Rule
        if 70 <= percentage <= 80:
            print("Accepted: Fill level is within 75%(+-5%),can go for packaging.")
        elif percentage > 80:
            print("Please lower the amount of liquid. Overfilled.")
        else:
            print("Fill till 75% otherwise won't be accepted.")
# Run this after training
upload_and_predict()

