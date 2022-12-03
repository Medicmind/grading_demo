#Convert a Medicmind grading classifier frozen_model.pb file into a keras tflite model
import tensorflow as tf
from tensorflow_addons.optimizers import LAMB
import glob
import sys
tf.keras.optimizers.Lamb = LAMB


if True:
  converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('camera/data/frozen_model.pb',
input_arrays = ['Reshape'],output_arrays = ['inception_v3/logits/logits/xw_plus_b'])

  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
  print('converting')
  tflite_model = converter.convert()

  print('writing')
  with open('pruned.lite', "wb") as f:
    f.write(tflite_model)
