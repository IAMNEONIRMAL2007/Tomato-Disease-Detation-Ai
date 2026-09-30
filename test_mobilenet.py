# Created By NirmalBorole
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Test MobileNetV2 with synthetic batch
x = np.random.uniform(0, 255, (32, 224, 224, 3)).astype(np.float32)
y = np.random.randint(0, 10, (32,)).astype(np.int32)

inputs = tf.keras.Input(shape=(224, 224, 3))
x_proc = preprocess_input(inputs)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x_proc)
base_model.trainable = False

x_pool = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x_dense = tf.keras.layers.Dense(256, activation='relu')(x_pool)
x_drop = tf.keras.layers.Dropout(0.4)(x_dense)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x_drop)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train 1 step
loss, acc = model.train_on_batch(x, y)
print(f"Step trained successfully: loss={loss:.4f}, acc={acc:.4f}")
