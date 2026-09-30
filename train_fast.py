# Created By NirmalBorole
"""
train_fast.py
High-speed Transfer Learning trainer using MobileNetV2.
Designed for rapid convergence and edge-ready deployment (.h5 and .tflite).
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'Dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("=" * 70)
print("🍅 HIGH-SPEED MOBILENETV2 TRANSFER LEARNING TRAINER")
print("=" * 70)

# Load dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode='int'
)

class_names = full_dataset.class_names
num_classes = len(class_names)
print(f"[+] Loaded {num_classes} classes from {DATA_DIR}")

# Split 70% train, 20% val, 10% test
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
train_batches = int(0.7 * total_batches)
val_batches = int(0.2 * total_batches)

train_ds = full_dataset.take(train_batches)
remaining = full_dataset.skip(train_batches)
val_ds = remaining.take(val_batches)

# Streaming prefetch without in-memory cache to stay well within RAM limits
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build Model
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = preprocess_input(inputs)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=x
)
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tomato_mobilenetv2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- STAGE 1: Feature Extraction (3 Epochs) ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1
)

print("\n--- STAGE 2: Fine-Tuning Top Layers (2 Epochs) ---")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1
)

# Save final models
final_h5 = os.path.join(MODELS_DIR, 'tomato_disease_final.h5')
model.save(final_h5)
print(f"\n[+] Saved final model: {final_h5}")

# Export TFLite
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODELS_DIR, 'tomato_disease.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"[+] Saved Quantized TFLite: {tflite_path} ({tflite_size_mb:.2f} MB)")
except Exception as e:
    print(f"[-] TFLite export warning: {e}")

print("\n🎉 Training & Export successfully completed!")
