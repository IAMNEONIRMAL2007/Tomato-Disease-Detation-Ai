# Created By NirmalBorole
import tensorflow as tf
import numpy as np

print("=" * 50)
print("TENSORFLOW STATUS:")
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
for gpu in tf.config.list_physical_devices('GPU'):
    print("GPU Details:", gpu)
print("=" * 50)
