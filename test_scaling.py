# Created By NirmalBorole
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
import numpy as np

# Test EfficientNetB0 preprocessing
x_raw = np.random.uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
x_scaled = x_raw / 255.0

eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
out_raw = eff(x_raw).numpy()
out_scaled = eff(x_scaled).numpy()

print(f"EfficientNet with [0, 255] input: mean={out_raw.mean():.4f}, std={out_raw.std():.4f}, max={out_raw.max():.4f}")
print(f"EfficientNet with [0, 1] input:   mean={out_scaled.mean():.4f}, std={out_scaled.std():.4f}, max={out_scaled.max():.4f}")
