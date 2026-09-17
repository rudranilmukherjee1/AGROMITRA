# ============================================================
# AGROMITRA - Master Pipeline (main.py)
# ============================================================
# Combines:
#   1. Trained MobileNetV2 Model (best_mobilenetv2.keras)
#   2. OpenCV Image Processing
#   3. AGROMITRA Decision Engine (decision_engine.py)
# ============================================================

import tensorflow as tf
import numpy as np
import cv2
from decision_engine import make_decision

# Configuration
IMAGE_SIZE = (224, 224)
MODEL_PATH = "best_mobilenetv2.keras"
IMAGE_PATH = "tomato_leaf.JPG"  # Change to any leaf image you want to test

# Class names must match your model's training order exactly
class_names = ["Early_Blight", "Healthy", "Late_Blight"]

print("Loading AGROMITRA AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load image using OpenCV
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"ERROR: Could not find or open image at '{IMAGE_PATH}'.")
    exit()

print(f"Image loaded successfully from: {IMAGE_PATH}")

# Preprocess image for MobileNetV2
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, IMAGE_SIZE)
image = image.astype(np.float32)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
image_array = np.expand_dims(image, axis=0)

# Run AI Prediction
predictions = model.predict(image_array, verbose=0)
scores = predictions[0]

predicted_index = np.argmax(scores)
predicted_disease = class_names[predicted_index]
confidence = float(scores[predicted_index])

# Simulated sensor values (Later these will come from your ESP32 hardware via Serial)
temperature = 29   # °C
humidity = 81      # %
soil_moisture = 34 # %

# Pass real AI outputs + environmental data into the Decision Engine
result = make_decision(
    disease=predicted_disease,
    confidence=confidence,
    temperature=temperature,
    humidity=humidity,
    soil_moisture=soil_moisture
)

# Display the Complete AGROMITRA System Report
print("\n" + "=" * 60)
print(" AGROMITRA INTEGRATED SYSTEM REPORT")
print("=" * 60)

print("AI VISION ANALYSIS")
print("-" * 60)
print(f"Predicted Disease : {result['disease']}")
print(f"Confidence Score  : {result['confidence_percentage']}% ({result['confidence_level']} Confidence)")
print("-" * 60)
print("All Class Probabilities:")
for name, score in zip(class_names, scores):
    print(f"  - {name:<12}: {score * 100:.2f}%")

print("\nENVIRONMENTAL METRICS (Sensors)")
print("-" * 60)
print(f"Temperature       : {result['temperature']} °C")
print(f"Humidity          : {result['humidity']} %")
print(f"Soil Moisture     : {result['soil_moisture']} %")
print(f"Environmental Risk: {result['environmental_risk']} (Score: {result['environmental_risk_score']})")

print("\nFINAL DECISION & ACTION")
print("-" * 60)
print(f"Calculated Severity: {result['severity']}")
print(f"Recommendation     : {result['recommendation']}")
print("=" * 60)