import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 1. SETTINGS
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 2. LOAD DATASETS
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_data.class_names
print("\nDetected Classes:", class_names)

# 3. PREPROCESSING PIPELINE
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_data = train_data.map(lambda x, y: (preprocess(x), y))
validation_data = validation_data.map(lambda x, y: (preprocess(x), y))
test_data = test_data.map(lambda x, y: (preprocess(x), y))

# 4. BUILD MODEL ARCHITECTURE
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. COMPUTE CLASS WEIGHTS (Fixes Healthy leaf misclassification)
train_labels = []
for _, labels in train_data:
    train_labels.extend(labels.numpy())

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights_dict = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weights_dict)

# 6. CALLBACKS & TRAINING
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_mobilenetv2.keras", monitor="val_loss", save_best_only=True)
]

print("\nStarting training...")
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# 7. FINAL EVALUATION
test_loss, test_accuracy = model.evaluate(test_data)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
print("Training complete! 🌱🔥")