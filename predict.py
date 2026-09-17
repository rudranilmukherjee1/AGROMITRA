import tensorflow as tf
import numpy as np
import cv2

IMAGE_SIZE = (224, 224)
MODEL_PATH = "best_mobilenetv2.keras"
IMAGE_PATH = "tomato_leaf.JPG"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

class_names = ["Early_Blight", "Healthy", "Late_Blight"]

# Load image using OpenCV
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"ERROR: OpenCV could not find or read '{IMAGE_PATH}'.")
    exit()

# Preprocess image for MobileNetV2
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, IMAGE_SIZE)
image = image.astype(np.float32)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
image_array = np.expand_dims(image, axis=0)

# Run Prediction
predictions = model.predict(image_array, verbose=0)
scores = predictions[0]

print("\n==============================")
print("AGROMITRA AI PREDICTION")
print("==============================")

for class_name, score in zip(class_names, scores):
    print(f"{class_name}: {score * 100:.2f}%")

predicted_index = np.argmax(scores)
predicted_class = class_names[predicted_index]
confidence = scores[predicted_index]

print("\nPredicted disease:", predicted_class)
print(f"Top score: {confidence * 100:.2f}%")